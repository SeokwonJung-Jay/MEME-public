"""DSPy Module wrapping MD-flat pipeline.

SIMBA optimizes the instruction (system prompt) for 3 Signatures:
  - IngestSig: session conversation → memory file update
  - RetrieveSig: question + memory file → relevant facts
  - AnswerSig: question + retrieved facts → final answer

The module runs a full episode end-to-end and returns all answers.
"""
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Dict

import dspy

# Add agents/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents.md_file import VirtualFileSystem, TOOLS, RETRIEVE_TOOLS, MEMORY_FILE


# ============================================================
# Tool-call JSONL logger (enabled via TOOL_CALL_LOG_PATH env var)
# ============================================================
# Writes one JSON object per tool call to the configured file. Thread-safe.
# Records every call made inside _run_tool_loop, including those made during
# SIMBA optimization (because every forward() pass goes through this path).

_TOOL_LOG_FH = None
_TOOL_LOG_LOCK = threading.Lock()


def _get_tool_log():
    global _TOOL_LOG_FH
    if _TOOL_LOG_FH is not None:
        return _TOOL_LOG_FH
    path = os.environ.get("TOOL_CALL_LOG_PATH")
    if not path:
        return None
    with _TOOL_LOG_LOCK:
        if _TOOL_LOG_FH is None:
            _TOOL_LOG_FH = open(path, "a", buffering=1)
    return _TOOL_LOG_FH


def _log_tool_call(tag: str, round_idx: int, tc, response_meta: dict):
    fh = _get_tool_log()
    if fh is None:
        return
    args = tc.function.arguments
    rec = {
        "ts": time.time(),
        "thread": threading.current_thread().name,
        "tag": tag,
        "round": round_idx,
        "tool_name": tc.function.name,
        "args_len": len(args),
        "args_md5": hashlib.md5(args.encode()).hexdigest()[:10],
        "args": args,
        **response_meta,
    }
    with _TOOL_LOG_LOCK:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# DSPy Signatures (instructions are optimized by SIMBA)
# ============================================================

class IngestSig(dspy.Signature):
    """You are a personal assistant with a persistent memory file (memory.md).
    After each conversation, save any information the user shared that may be useful in future sessions.
    Keep it compact — one fact per line with timestamp [YYYY/MM/DD].
    If information has changed, update it. If something was removed or cancelled, remove the old entry.
    Do NOT save conversation summaries, assistant responses, or temporary task state.
    Read the conversation session below and update your memory accordingly using the provided tools."""

    conversation = dspy.InputField(desc="The conversation session to process")
    tool_calls = dspy.OutputField(desc="Tool calls to execute (read_memory / write_memory / append_memory)")


class RetrieveSig(dspy.Signature):
    """You have access to read_memory() to look up information about the user.
    Read your memory file and extract ONLY the facts relevant to answering the question.
    Return relevant facts as-is (do not rephrase or summarize). If nothing is relevant, say '(no relevant facts)'."""

    question = dspy.InputField(desc="The question to answer")
    relevant_facts = dspy.OutputField(desc="Relevant facts from memory, or '(no relevant facts)'")


# Answer instruction is a fixed constant — NOT a dspy.Predict, NOT an optimization
# target. Reason: the answer LM is Sonnet, which does not support the `seed`
# parameter, so its output is non-deterministic at temp=0 and any SIMBA "advice"
# against it would be optimizing pure noise. Keep the baseline instruction frozen
# so the evaluation signal reflects ingest/retrieve prompt changes only.
ANSWER_INSTRUCTION = """Answer the user's question based ONLY on the context provided below.
If the information is not in the context, say you don't have that information.
Answer with ONLY the value. Do not explain or add context."""


# ============================================================
# Tool loop helpers (use DSPy LM under the hood)
# ============================================================

_OAI_CLIENT = None


def _get_openai_client():
    global _OAI_CLIENT
    if _OAI_CLIENT is None:
        from openai import OpenAI
        _OAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _OAI_CLIENT


def _run_tool_loop(lm, instruction: str, user_msg: str,
                  vfs: VirtualFileSystem, tools: List[Dict],
                  max_rounds: int = 5, model: str = "gpt-4.1-mini",
                  log_tag: str = "") -> str:
    """OpenAI function calling loop using raw OpenAI SDK (bypasses DSPy LM for tools).

    Uses `model` for the task. Returns the final assistant text.
    When TOOL_CALL_LOG_PATH env var is set, every tool call is appended as JSONL.
    """
    client = _get_openai_client()
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_msg},
    ]
    final_text = ""
    instr_md5 = hashlib.md5(instruction.encode()).hexdigest()[:10]

    seed = int(os.environ.get("OPENAI_SEED", "42"))
    # Explicit short timeout so a dead connection fails fast instead of
    # hanging the SIMBA worker thread indefinitely (see stuck-run post-mortem).
    call_timeout = float(os.environ.get("OPENAI_CALL_TIMEOUT_SEC", "120"))
    for round_idx in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            max_tokens=4000,  # headroom to avoid mid-JSON truncation; output billed only if used
            seed=seed,
            timeout=call_timeout,
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, 'tool_calls', None)

        response_meta = {
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": response.choices[0].finish_reason,
            "instr_len": len(instruction),
            "instr_md5": instr_md5,
        }

        if not tool_calls:
            final_text = msg.content or ""
            break

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            _log_tool_call(log_tag, round_idx, tc, response_meta)
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                # Model truncated its JSON output (usually max_tokens cutoff).
                # Feed the error back as the tool result; model can self-correct
                # in the next round. This is distinct from a timeout/network
                # failure: malformed args are stochastic and recoverable.
                import logging
                logging.warning(
                    f"[_run_tool_loop] malformed tool_call args "
                    f"(finish_reason={response_meta.get('finish_reason')}, "
                    f"args_len={len(tc.function.arguments)}): {e}. Feeding error back."
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: Could not parse arguments for "
                               f"{tc.function.name} ({e}). Retry with shorter content.",
                })
                continue
            result = vfs.execute_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return final_text


# ============================================================
# MD-flat DSPy Module
# ============================================================

class MDFlatProgram(dspy.Module):
    """Full MD-flat episode pipeline as a DSPy Module.

    Uses 2 LMs (matching production setup):
      - task_lm (default from dspy.settings.lm): ingest + retrieve (gpt-4.1-mini)
      - answer_lm (passed in): final answer phase (Sonnet)

    SIMBA optimizes the instruction field of 2 sub-modules: ingest + retrieve.
    The answer instruction is a fixed constant (see ANSWER_INSTRUCTION) because
    Sonnet does not support the `seed` parameter, so any SIMBA advice on it
    would be optimizing noise.
    """

    def __init__(self, answer_lm=None, task_model: str = "gpt-4.1-mini",
                 answer_model: str = "claude-sonnet-4-20250514"):
        super().__init__()
        self.ingest = dspy.Predict(IngestSig)
        self.retrieve = dspy.Predict(RetrieveSig)
        # answer is intentionally NOT a dspy.Predict so named_predictors() skips it
        self.answer_lm = answer_lm
        self.task_model = task_model  # OpenAI model name for tool calling
        self.answer_model = answer_model  # Anthropic model for answer phase

    def forward(self, episode_id: int, domain: str, sessions: List[Dict],
                before_pos: int, after_pos: int,
                before_questions: List[Dict], after_questions: List[Dict]):
        """Run full episode: phase1 → before questions → phase2 → after questions."""
        import logging
        vfs = VirtualFileSystem()
        lm = dspy.settings.lm
        answer_lm = self.answer_lm or lm
        logging.info(f"[{domain} ep{episode_id}] START — task_model={self.task_model}, "
                     f"answer_model={self.answer_model}, "
                     f"ingest_instr_len={len(self.ingest.signature.instructions)}")

        # Ingest/retrieve are SIMBA-optimized; answer is a frozen constant.
        ingest_instruction = self.ingest.signature.instructions
        retrieve_instruction = self.retrieve.signature.instructions
        answer_instruction = ANSWER_INSTRUCTION

        ep_tag = f"{domain}_{episode_id}"
        run_uid = f"{os.getpid()}_{int(time.time()*1e6)}_{threading.get_ident()}"

        # Phase 1: Feed sessions up to before-questions
        for sidx, sess in enumerate(sessions[:before_pos]):
            conv_text = f"[Session: {sess.get('timestamp', 'unknown')}]\n"
            for turn in sess["conversation"]:
                role = "User" if turn["role"] == "user" else "Assistant"
                conv_text += f"{role}: {turn['content']}\n"
            _run_tool_loop(lm, ingest_instruction, conv_text, vfs, TOOLS,
                           model=self.task_model,
                           log_tag=f"{run_uid}|{ep_tag}|phase1|ingest|sess{sidx+1}")

        mem_len_before = len(vfs.get_all())
        logging.info(f"[{domain} ep{episode_id}] Phase 1 done. Memory: {mem_len_before} chars. "
                     f"Sample: {vfs.get_all()[:200]}")
        before_answers = self._answer_questions(
            lm, answer_lm, before_questions, vfs, retrieve_instruction, answer_instruction,
            run_uid=run_uid, ep_tag=ep_tag, phase="phase1"
        )

        # Phase 2: Feed remaining sessions
        for sidx, sess in enumerate(sessions[before_pos:after_pos]):
            conv_text = f"[Session: {sess.get('timestamp', 'unknown')}]\n"
            for turn in sess["conversation"]:
                role = "User" if turn["role"] == "user" else "Assistant"
                conv_text += f"{role}: {turn['content']}\n"
            _run_tool_loop(lm, ingest_instruction, conv_text, vfs, TOOLS,
                           model=self.task_model,
                           log_tag=f"{run_uid}|{ep_tag}|phase2|ingest|sess{before_pos+sidx+1}")

        mem_len_after = len(vfs.get_all())
        logging.info(f"[{domain} ep{episode_id}] Phase 2 done. Memory: {mem_len_after} chars "
                     f"(Δ{mem_len_after - mem_len_before})")
        after_answers = self._answer_questions(
            lm, answer_lm, after_questions, vfs, retrieve_instruction, answer_instruction,
            run_uid=run_uid, ep_tag=ep_tag, phase="phase2"
        )

        return dspy.Prediction(
            episode_id=episode_id,
            domain=domain,
            before_answers=before_answers,
            after_answers=after_answers,
        )

    def _answer_questions(self, lm, answer_lm, questions, vfs, retrieve_instr, answer_instr,
                          run_uid: str = "", ep_tag: str = "", phase: str = ""):
        import logging
        out = []
        for qi, q in enumerate(questions):
            # Retrieve (task LM — gpt-4.1-mini)
            retrieve_prompt = f"Question: {q['question']}"
            facts = _run_tool_loop(lm, retrieve_instr, retrieve_prompt,
                                   vfs, RETRIEVE_TOOLS, model=self.task_model,
                                   log_tag=f"{run_uid}|{ep_tag}|{phase}|retrieve|q{qi+1}")
            logging.debug(f"[Q{qi}] retrieved ({len(facts)} chars): {facts[:100]}...")

            # Answer (answer LM — Sonnet). Explicit timeout via litellm request
            # timeout; Anthropic API has no seed support but we still fail-fast.
            answer_prompt = f"Context:\n{facts}\n\nQuestion: {q['question']}"
            ans_timeout = float(os.environ.get("ANSWER_CALL_TIMEOUT_SEC", "120"))
            ans_response = answer_lm(
                messages=[
                    {"role": "system", "content": answer_instr},
                    {"role": "user", "content": answer_prompt},
                ],
                temperature=0,
                max_tokens=500,
                timeout=ans_timeout,
            )
            if isinstance(ans_response, list) and ans_response:
                ans_response = ans_response[0]
            if hasattr(ans_response, 'choices'):
                answer = ans_response.choices[0].message.content or ""
            else:
                answer = str(ans_response)

            out.append({
                "task_type": q.get("task_type", ""),
                "entity": q.get("entity", []),
                "entity_values": q.get("entity_values", {}),
                "question": q["question"],
                "expected_answer": q.get("expected_answer", q.get("gold_answer", "")),
                "agent_answer": answer,
                "retrieved_context": facts,
            })
        return out
