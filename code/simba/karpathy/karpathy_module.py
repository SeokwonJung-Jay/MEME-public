"""DSPy Module wrapping Karpathy Wiki pipeline.

SIMBA optimizes 3 Signatures:
  - FlushSig: session conversation → daily log entry
  - CompileSig: daily log + wiki → updated knowledge articles (tool-using)
  - QuerySig: question + wiki → answer (tool-using, read-only)

The 3 default instructions are extracted from the original
.deps/claude-memory-compiler/scripts/{flush,compile,query}.py prompts,
split so task-instruction text lives in the system prompt (SIMBA-optimizable)
and schema/wiki/variable data live in the user message.

No modifications to .deps/claude-memory-compiler/.
All 3 phases run on gpt-4.1-mini (matching Karpathy's new native-LLM convention).
Karpathy has no separate "answer LM" — query output IS the answer.
"""
import os
import sys
import shutil
import time
from pathlib import Path
from typing import List, Dict

import dspy

# Import openai_tool_agent from the monorepo (read-only — no edits to .deps/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from agents.openai_tool_agent import run_tool_agent, ALL_TOOLS, READ_ONLY_TOOLS

_KARPATHY_ROOT = _REPO_ROOT / ".deps" / "claude-memory-compiler"
_AGENTS_MD_SRC = _KARPATHY_ROOT / "AGENTS.md"


# ============================================================
# DSPy Signatures (instructions optimized by SIMBA)
# ============================================================

class FlushSig(dspy.Signature):
    """Review the conversation context and respond with a concise summary of important
    items that should be preserved in the daily log.
    Do NOT use any tools — just return plain text.

    Format your response as a structured daily log entry with these sections:

    **Context:** [One line about what the user was working on]

    **Key Exchanges:**
    - [Important Q&A or discussions]

    **Decisions Made:**
    - [Any decisions with rationale]

    **Lessons Learned:**
    - [Gotchas, patterns, or insights discovered]

    **Action Items:**
    - [Follow-ups or TODOs mentioned]

    Skip anything that is routine, trivial, or obvious.
    Only include sections that have actual content. If nothing is worth saving,
    respond with exactly: FLUSH_OK"""

    conversation = dspy.InputField(desc="The conversation context to flush")
    daily_entry = dspy.OutputField(desc="Structured daily log entry or 'FLUSH_OK'")


class CompileSig(dspy.Signature):
    """You are a knowledge compiler. Your job is to read a daily conversation log
    and extract knowledge into structured wiki articles.

    Rules:
    1. Extract key concepts — identify 3-7 distinct concepts worth their own article.
    2. Create concept articles in knowledge/concepts/ — one .md file per concept,
       using the exact article format from AGENTS.md (YAML frontmatter + sections),
       with sources: pointing to the daily log file and [[concepts/slug]] wikilinks.
    3. Create connection articles in knowledge/connections/ if the log reveals non-obvious
       relationships between 2+ existing concepts.
    4. Update existing articles if the log adds new information to them; add the new
       source to frontmatter.
    5. Update knowledge/index.md — append new entries as table rows.
    6. Append to knowledge/log.md — a timestamped entry listing articles created/updated.

    Quality standards:
    - Complete YAML frontmatter on every article.
    - Every article links to at least 2 others via [[wikilinks]].
    - Key Points: 3-5 bullets. Details: 2+ paragraphs. Related Concepts: 2+ entries.

    Use the provided file tools (read_file, write_file, edit_file, glob_files, grep_files)
    to create and update articles."""

    task_input = dspy.InputField(desc="Schema (AGENTS.md), wiki index, existing articles, daily log to compile, and file path targets")
    status = dspy.OutputField(desc="Completion status")


class QuerySig(dspy.Signature):
    """You are a knowledge base query engine. Answer the user's question by consulting
    the knowledge base.

    How to answer:
    1. Read the INDEX section first — it lists every article with a one-line summary.
    2. Identify 3-10 articles relevant to the question.
    3. Read those articles carefully (they're included below).
    4. Synthesize a clear, thorough answer.
    5. Cite sources using [[wikilinks]] (e.g., [[concepts/supabase-auth]]).
    6. If the knowledge base doesn't contain relevant info, say so honestly.

    Answer with ONLY the value. Do not explain or add context unless the question asks for it."""

    knowledge_base = dspy.InputField(desc="Wiki content with index and articles")
    question = dspy.InputField(desc="The user's question")
    answer = dspy.OutputField(desc="Concise answer")


# ============================================================
# Workspace (isolated per-episode knowledge dir on disk)
# ============================================================

class Workspace:
    """Per-episode Karpathy workspace. Mirrors the layout of .deps/claude-memory-compiler
    (daily/, knowledge/{concepts,connections,qa}/, index.md, log.md, AGENTS.md)."""

    def __init__(self, root_dir: Path):
        self.root = root_dir
        self.daily = root_dir / "daily"
        self.knowledge = root_dir / "knowledge"
        self.concepts = self.knowledge / "concepts"
        self.connections = self.knowledge / "connections"
        self.qa = self.knowledge / "qa"
        self.index_file = self.knowledge / "index.md"
        self.log_file = self.knowledge / "log.md"
        for d in (self.daily, self.concepts, self.connections, self.qa):
            d.mkdir(parents=True, exist_ok=True)
        # Copy AGENTS.md schema
        self.agents_md = root_dir / "AGENTS.md"
        if _AGENTS_MD_SRC.exists() and not self.agents_md.exists():
            self.agents_md.write_text(_AGENTS_MD_SRC.read_text(encoding="utf-8"), encoding="utf-8")

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def read_wiki_index(self) -> str:
        if self.index_file.exists():
            return self.index_file.read_text(encoding="utf-8")
        return "(empty)"

    def list_wiki_articles(self) -> List[Path]:
        articles = []
        for d in (self.concepts, self.connections, self.qa):
            articles.extend(sorted(d.glob("*.md")))
        return articles

    def existing_articles_block(self) -> str:
        parts = []
        for art in self.list_wiki_articles():
            rel = art.relative_to(self.knowledge)
            parts.append(f"### {rel}\n```markdown\n{art.read_text(encoding='utf-8')}\n```")
        return "\n\n".join(parts) if parts else "(No existing articles yet)"

    def read_all_wiki_content(self) -> str:
        parts = []
        if self.index_file.exists():
            parts.append(f"## INDEX\n\n{self.index_file.read_text(encoding='utf-8')}")
        for art in self.list_wiki_articles():
            rel = art.relative_to(self.knowledge)
            parts.append(f"### {rel}\n\n{art.read_text(encoding='utf-8')}")
        return "\n\n".join(parts) if parts else "(empty knowledge base)"


# ============================================================
# LLM helpers
# ============================================================

_OAI_CLIENT = None


def _get_openai_client():
    global _OAI_CLIENT
    if _OAI_CLIENT is None:
        from openai import OpenAI
        _OAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _OAI_CLIENT


def _run_flush_llm(system_prompt: str, conversation: str, model: str) -> str:
    client = _get_openai_client()
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"## Conversation Context\n\n{conversation}"},
        ],
        temperature=0,
        max_tokens=2000,
    )
    seed_env = os.environ.get("OPENAI_SEED")
    if seed_env is not None:
        kwargs["seed"] = int(seed_env)
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ============================================================
# DSPy Module
# ============================================================

class KarpathyProgram(dspy.Module):
    """Full Karpathy Wiki episode pipeline as a DSPy Module.

    SIMBA will optimize the instruction field of all 3 sub-modules
    (flush / compile / query). Uses gpt-4.1-mini for all phases.
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        super().__init__()
        self.flush = dspy.Predict(FlushSig)
        self.compile = dspy.Predict(CompileSig)
        self.query = dspy.Predict(QuerySig)
        self.model = model

    def _make_workspace(self, episode_id: int, domain: str) -> Workspace:
        ws_root = (_REPO_ROOT / ".workspaces" / "simba_karpathy"
                   / f"{os.getpid()}_{int(time.time()*1e6)}_{domain}_{episode_id}")
        return Workspace(ws_root)

    def _run_flush(self, conversation: str, ws: Workspace,
                   session_idx: int, timestamp: str) -> bool:
        """Returns True if content was saved, False if FLUSH_OK/error."""
        system_prompt = self.flush.signature.instructions
        result = _run_flush_llm(system_prompt, conversation, self.model)
        if not result or result.strip() == "FLUSH_OK" or "FLUSH_ERROR" in result:
            return False
        date_str = (timestamp.split('(')[0].strip().replace('/', '-')
                    if timestamp else f"session-{session_idx}")
        daily_file = ws.daily / f"{date_str}.md"
        with open(daily_file, "a", encoding="utf-8") as f:
            f.write(f"\n---\n## Session {session_idx} [{timestamp}]\n{result}\n")
        return True

    def _run_compile(self, ws: Workspace, log_prefix: str = ""):
        """Compile all daily logs → wiki articles via OpenAI tool agent."""
        system_prompt = self.compile.signature.instructions
        schema = ws.agents_md.read_text(encoding="utf-8") if ws.agents_md.exists() else ""
        wiki_index = ws.read_wiki_index()
        existing = ws.existing_articles_block()

        for daily_file in sorted(ws.daily.glob("*.md")):
            log_content = daily_file.read_text(encoding="utf-8")
            user_msg = f"""## Schema (AGENTS.md)

{schema}

## Current Wiki Index

{wiki_index}

## Existing Wiki Articles

{existing}

## Daily Log to Compile

**File:** {daily_file.name}

{log_content}

## File paths

- Concept articles → {ws.concepts}
- Connection articles → {ws.connections}
- Index → {ws.index_file}
- Log → {ws.log_file}

Use the file tools to read existing articles and write new ones.
"""
            run_tool_agent(
                prompt=user_msg,
                cwd=str(ws.root),
                model=self.model,
                system_prompt=system_prompt,
                tools=ALL_TOOLS,
                max_turns=30,
                log_tag=f"{log_prefix}|compile|daily={daily_file.name}",
            )

    def _run_query(self, question: str, ws: Workspace, log_prefix: str = "",
                   qi: int = 0) -> str:
        """Query the wiki; returns answer string (directly used as agent_answer)."""
        system_prompt = self.query.signature.instructions
        wiki_content = ws.read_all_wiki_content()
        user_msg = f"""## Knowledge Base

{wiki_content}

## Question

{question}
"""
        answer = run_tool_agent(
            prompt=user_msg,
            cwd=str(ws.root),
            model=self.model,
            system_prompt=system_prompt,
            tools=READ_ONLY_TOOLS,
            max_turns=15,
            log_tag=f"{log_prefix}|query|q{qi+1}",
        )
        return answer.strip() if answer else "(empty)"

    def _answer_questions(self, questions: List[Dict], ws: Workspace,
                          log_prefix: str = "") -> List[Dict]:
        out = []
        for qi, q in enumerate(questions):
            ans = self._run_query(q["question"], ws, log_prefix=log_prefix, qi=qi)
            out.append({
                "task_type": q.get("task_type", ""),
                "entity": q.get("entity", []),
                "entity_values": q.get("entity_values", {}),
                "question": q["question"],
                "expected_answer": q.get("expected_answer", q.get("gold_answer", "")),
                "agent_answer": ans,
                "retrieved_context": "",
            })
        return out

    def forward(self, episode_id: int, domain: str, sessions: List[Dict],
                before_pos: int, after_pos: int,
                before_questions: List[Dict], after_questions: List[Dict]):
        import logging
        ws = self._make_workspace(episode_id, domain)
        logging.info(f"[karpathy {domain} ep{episode_id}] START — model={self.model}, "
                     f"flush_instr_len={len(self.flush.signature.instructions)}, "
                     f"compile_instr_len={len(self.compile.signature.instructions)}, "
                     f"query_instr_len={len(self.query.signature.instructions)}, "
                     f"ws={ws.root}")

        ep_tag = f"{domain}_{episode_id}"
        run_uid = f"{os.getpid()}_{int(time.time()*1e6)}_{id(ws)}"
        log_p1 = f"{run_uid}|{ep_tag}|phase1"
        log_p2 = f"{run_uid}|{ep_tag}|phase2"

        try:
            # Phase 1: flush sessions up to before_pos
            any_saved_1 = False
            for i, sess in enumerate(sessions[:before_pos]):
                conv_text = f"[Session: {sess.get('timestamp', 'unknown')}]\n"
                for turn in sess["conversation"]:
                    role = "User" if turn["role"] == "user" else "Assistant"
                    conv_text += f"{role}: {turn['content']}\n"
                saved = self._run_flush(conv_text, ws, i + 1, sess.get('timestamp', ''))
                any_saved_1 = any_saved_1 or saved

            if any_saved_1:
                logging.info(f"[karpathy {domain} ep{episode_id}] Phase 1 compile (daily={len(list(ws.daily.glob('*.md')))})")
                self._run_compile(ws, log_prefix=log_p1)
                logging.info(f"[karpathy {domain} ep{episode_id}] Phase 1 compile done — articles={len(ws.list_wiki_articles())}")
            else:
                logging.info(f"[karpathy {domain} ep{episode_id}] Phase 1: nothing to compile (all FLUSH_OK)")

            before_answers = self._answer_questions(before_questions, ws, log_prefix=log_p1)

            # Phase 2: flush sessions[before_pos:after_pos]
            any_saved_2 = False
            for i, sess in enumerate(sessions[before_pos:after_pos]):
                conv_text = f"[Session: {sess.get('timestamp', 'unknown')}]\n"
                for turn in sess["conversation"]:
                    role = "User" if turn["role"] == "user" else "Assistant"
                    conv_text += f"{role}: {turn['content']}\n"
                saved = self._run_flush(conv_text, ws, before_pos + i + 1, sess.get('timestamp', ''))
                any_saved_2 = any_saved_2 or saved

            if any_saved_2:
                logging.info(f"[karpathy {domain} ep{episode_id}] Phase 2 compile")
                self._run_compile(ws, log_prefix=log_p2)
                logging.info(f"[karpathy {domain} ep{episode_id}] Phase 2 compile done — articles={len(ws.list_wiki_articles())}")
            else:
                logging.info(f"[karpathy {domain} ep{episode_id}] Phase 2: nothing to compile")

            after_answers = self._answer_questions(after_questions, ws, log_prefix=log_p2)

            return dspy.Prediction(
                episode_id=episode_id,
                domain=domain,
                before_answers=before_answers,
                after_answers=after_answers,
            )
        finally:
            # Free disk to avoid buildup across many SIMBA candidates × episodes.
            # Comment out to keep workspaces for post-mortem debugging.
            ws.cleanup()
