"""DSPy Module wrapping Mem0 episode pipeline.

SIMBA optimizes 1 Signature:
  - ExtractionSig → ADDITIVE_EXTRACTION_PROMPT (mem0 v2's only on-path prompt)

mem0 v2 invokes ADDITIVE_EXTRACTION_PROMPT inline at extraction time. We
override it via mem0_prompts_override.MemoryPromptOverride: a per-Memory
patch on llm.generate_response that swaps in the current Signature
instructions before each extraction call.

Parallelization: each SIMBA worker thread gets its own Memory + Qdrant
collection (via thread-local), and its own MemoryPromptOverride. Threads
do not share llm.generate_response, so candidate prompts cannot collide.

Answer LM: gpt-4.1-mini (matches MD-flat new convention). Mem0's
m.search() is the retrieval; the unified answer step happens after.
"""
import asyncio
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List

# Make the main release's agents/ importable so we can reuse AnthropicAsOpenAI
# for the answer phase when --answer-model is a Claude model.
_RELEASE_CODE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_RELEASE_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RELEASE_CODE_ROOT))

import dspy

from mem0_prompts_override import (
    DEFAULT_PROMPT,
    MemoryPromptOverride,
)


# ============================================================
# DSPy Signature (instructions = mem0's default at baseline)
# ============================================================

class ExtractionSig(dspy.Signature):
    __doc__ = DEFAULT_PROMPT

    conversation = dspy.InputField(desc="New conversation messages plus prior context (existing memories, observation date, etc.)")
    extracted = dspy.OutputField(desc="JSON object with key 'memory' listing extracted facts as ADD operations")


# ============================================================
# Per-thread Memory + Qdrant collection isolation
# ============================================================

_tl = threading.local()


def _get_thread_memory(internal_model: str, qdrant_host: str, qdrant_port: int):
    """Return (Memory, MemoryPromptOverride, collection_name) for this thread.
    Lazily creates them on first call. Reuses across episodes."""
    if hasattr(_tl, "memory"):
        return _tl.memory, _tl.override, _tl.collection_name

    from mem0 import Memory

    collection_name = f"simba_mem0_{threading.get_ident()}_{uuid.uuid4().hex[:8]}"
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": qdrant_host,
                "port": qdrant_port,
                "collection_name": collection_name,
                "embedding_model_dims": 1536,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": internal_model,
                "api_key": os.environ.get("OPENAI_API_KEY"),
            },
        },
    }
    memory = Memory.from_config(config)
    override = MemoryPromptOverride(memory)
    _tl.memory = memory
    _tl.override = override
    _tl.collection_name = collection_name

    import logging
    logging.info(f"[mem0 thread {threading.current_thread().name}] collection={collection_name}")
    return memory, override, collection_name


def _reset_thread_collection(qdrant_host: str, qdrant_port: int, collection_name: str):
    """Drop and recreate the collection so each episode starts fresh."""
    from qdrant_client import QdrantClient
    qc = QdrantClient(host=qdrant_host, port=qdrant_port)
    try:
        qc.delete_collection(collection_name)
    except Exception:
        pass


# ============================================================
# Answer-phase LLM (separate, mirrors MD-flat convention)
# ============================================================

ANSWER_INSTRUCTION = """Answer the user's question based ONLY on the context provided below.
If the information is not in the context, say you don't have that information.
Answer with ONLY the value. Do not explain or add context."""


_OAI_CLIENT = None
_ANTH_CLIENT = None


def _get_answer_client(model: str):
    """Return the right SDK client for the answer LM. Claude models are
    routed through agents/anthropic_adapter.AnthropicAsOpenAI so the call
    site can keep the OpenAI chat.completions.create shape."""
    global _OAI_CLIENT, _ANTH_CLIENT
    if model.startswith("claude"):
        if _ANTH_CLIENT is None:
            from agents.anthropic_adapter import AnthropicAsOpenAI
            _ANTH_CLIENT = AnthropicAsOpenAI(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return _ANTH_CLIENT
    if _OAI_CLIENT is None:
        from openai import OpenAI
        _OAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _OAI_CLIENT


def _answer_via_llm(question: str, context: str, model: str) -> str:
    client = _get_answer_client(model)
    timeout = float(os.environ.get("ANSWER_CALL_TIMEOUT_SEC", "120"))
    create_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_INSTRUCTION},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=500,
        timeout=timeout,
    )
    if not model.startswith("claude"):
        # Anthropic API does not expose a public seed parameter.
        create_kwargs["seed"] = int(os.environ.get("OPENAI_SEED", "42"))
    resp = client.chat.completions.create(**create_kwargs)
    return (resp.choices[0].message.content or "").strip()


# ============================================================
# DSPy Module
# ============================================================

class Mem0Program(dspy.Module):
    """Full Mem0 episode pipeline. SIMBA optimizes ExtractionSig only."""

    def __init__(self,
                 internal_model: str = "gpt-4.1-mini",
                 answer_model: str = "gpt-4.1-mini",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333):
        super().__init__()
        self.extract = dspy.Predict(ExtractionSig)
        self.internal_model = internal_model
        self.answer_model = answer_model
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

    def _ingest_sessions(self, memory, sessions, user_id, start_idx=0):
        for i, sess in enumerate(sessions):
            conv_text = f"[Session: {sess.get('timestamp', 'unknown')}]\n"
            for turn in sess["conversation"]:
                role = "User" if turn['role'] == 'user' else "Assistant"
                conv_text += f"{role}: {turn['content']}\n"
            try:
                memory.add(conv_text, user_id=user_id)
            except Exception as e:
                import logging
                logging.warning(f"[mem0] add() failed at session {start_idx+i+1}: {e}")

    def _search(self, memory, question: str, user_id: str) -> str:
        try:
            results = memory.search(query=question, filters={"user_id": user_id})
        except Exception as e:
            import logging
            logging.warning(f"[mem0] search() failed: {e}")
            return "(no relevant facts)"
        memories = results.get("results", [])
        if not memories:
            return "(no relevant facts)"
        lines = []
        for mem in memories:
            text = mem.get("memory", "")
            score = mem.get("score", 0)
            lines.append(f"[score={score:.2f}] {text}")
        return "\n".join(lines)

    def _answer_questions(self, memory, questions, user_id):
        out = []
        for q in questions:
            facts = self._search(memory, q['question'], user_id)
            ans = _answer_via_llm(q['question'], facts, self.answer_model)
            out.append({
                "task_type": q.get("task_type", ""),
                "entity": q.get("entity", []),
                "entity_values": q.get("entity_values", {}),
                "question": q['question'],
                "expected_answer": q.get("expected_answer", q.get("gold_answer", "")),
                "agent_answer": ans,
                "retrieved_context": facts,
            })
        return out

    def forward(self, episode_id: int, domain: str, sessions: List[Dict],
                before_pos: int, after_pos: int,
                before_questions: List[Dict], after_questions: List[Dict]):
        import logging

        memory, override, collection_name = _get_thread_memory(
            self.internal_model, self.qdrant_host, self.qdrant_port,
        )

        # Reset collection for fresh episode (drops all data from previous use).
        _reset_thread_collection(self.qdrant_host, self.qdrant_port, collection_name)
        # Recreate Memory because Qdrant collection was dropped (mem0 caches state).
        from mem0 import Memory
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": self.qdrant_host,
                    "port": self.qdrant_port,
                    "collection_name": collection_name,
                    "embedding_model_dims": 1536,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.internal_model,
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                },
            },
        }
        memory = Memory.from_config(config)
        override = MemoryPromptOverride(memory)
        _tl.memory = memory
        _tl.override = override

        # Apply current candidate prompt
        prompt = self.extract.signature.instructions
        override.set_prompt(prompt)

        user_id = f"simba_{domain}_{episode_id}_{uuid.uuid4().hex[:8]}"

        logging.info(f"[mem0 {domain} ep{episode_id}] START — collection={collection_name}, "
                     f"user_id={user_id}, extract_instr_len={len(prompt)}")

        try:
            t0 = time.time()
            self._ingest_sessions(memory, sessions[:before_pos], user_id, start_idx=0)
            logging.info(f"[mem0 {domain} ep{episode_id}] Phase 1 ingest done in {time.time()-t0:.1f}s")

            before_answers = self._answer_questions(memory, before_questions, user_id)

            t0 = time.time()
            self._ingest_sessions(memory, sessions[before_pos:after_pos], user_id,
                                  start_idx=before_pos)
            logging.info(f"[mem0 {domain} ep{episode_id}] Phase 2 ingest done in {time.time()-t0:.1f}s")

            after_answers = self._answer_questions(memory, after_questions, user_id)

            return dspy.Prediction(
                episode_id=episode_id,
                domain=domain,
                before_answers=before_answers,
                after_answers=after_answers,
            )
        finally:
            # Drop the collection so the next forward() call starts clean
            _reset_thread_collection(self.qdrant_host, self.qdrant_port, collection_name)
