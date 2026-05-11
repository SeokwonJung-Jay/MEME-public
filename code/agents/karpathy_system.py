"""
KarpathyWikiMemory — Memory system using claude-memory-compiler (Karpathy Wiki style).

Architecture:
  - Ingest: session transcript → flush.py (Claude Agent SDK extracts knowledge) → daily log
  - After all sessions: compile.py → daily logs → knowledge articles (concepts/, connections/, qa/)
  - Retrieve: query.py → reads index.md + articles → returns relevant knowledge
  - Answer: unified LLM (from base) uses retrieved context

Transcript-based: session text goes directly to flush.py. No agent-mediated ingest.
Uses official claude-memory-compiler code (no modifications).

Requires: claude-memory-compiler cloned at code/.deps/claude-memory-compiler/
  git clone https://github.com/coleam00/claude-memory-compiler code/.deps/claude-memory-compiler
"""

import os
import sys
import asyncio
import importlib
import inspect
from pathlib import Path

KARPATHY_PATH = os.path.join(os.path.dirname(__file__), "..", ".deps", "claude-memory-compiler")
KARPATHY_SCRIPTS = os.path.join(KARPATHY_PATH, "scripts")
if KARPATHY_PATH not in sys.path:
    sys.path.insert(0, KARPATHY_PATH)
if KARPATHY_SCRIPTS not in sys.path:
    sys.path.insert(0, KARPATHY_SCRIPTS)


def _supported_kwargs(fn, **kwargs):
    """Drop kwargs not accepted by fn. Some upstream versions of
    run_flush / compile_daily_log / run_query do not expose `model=`,
    so we introspect the signature and only pass kwargs the local
    clone accepts."""
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

from agents.base import BaseMemorySystem


class KarpathyWikiMemory(BaseMemorySystem):
    """Karpathy Wiki style memory system.

    Transcript-based ingest: session → flush.py → daily log.
    Compile deferred to retrieve time (2× per episode: before before-questions
    and before after-questions), matching the original design intent of
    compiling accumulated logs rather than per-session.
    """

    def __init__(self, model="claude-sonnet-4-20250514", api_key=None, internal_model=None, **kwargs):
        # Internal LLM for flush/compile/query
        # Default: Haiku. Override with --internal-model.
        # API keys are read directly from env by the upstream
        # claude-memory-compiler scripts; nothing in this class stores them.
        self._internal_model = internal_model or "claude-haiku-4-5-20251001"
        self._work_dir = None
        self._last_retrieved_context = ""
        self._session_count = 0
        self._needs_compile = False
        # Persistent compile state across 1st and 2nd compile calls in the same
        # episode. compile_daily_log writes {"ingested": {filename: {hash, ...}}}
        # into this dict; _compile() skips daily files whose hash hasn't changed
        # since last compile — avoids re-processing unchanged daily logs during
        # finalize_ingest (which otherwise reprocesses everything from scratch).
        self._compile_state = {"ingested": {}}
        self._init_workspace()

    def _init_workspace(self):
        """Create isolated workspace for this episode."""
        import time
        workspaces_dir = Path(__file__).resolve().parent.parent / ".workspaces" / "karpathy"
        self._work_dir = workspaces_dir / f"{os.getpid()}_{int(time.time())}"
        self._work_dir.mkdir(parents=True, exist_ok=True)
        (self._work_dir / "daily").mkdir(exist_ok=True)
        (self._work_dir / "knowledge").mkdir(exist_ok=True)
        (self._work_dir / "knowledge" / "concepts").mkdir(exist_ok=True)
        (self._work_dir / "knowledge" / "connections").mkdir(exist_ok=True)
        (self._work_dir / "knowledge" / "qa").mkdir(exist_ok=True)
        (self._work_dir / "scripts").mkdir(exist_ok=True)

        # Override config paths to use our workspace
        os.environ["KARPATHY_ROOT"] = str(self._work_dir)

        # Symlink AGENTS.md so compile.py can read the schema from workspace
        agents_src = Path(KARPATHY_PATH) / "AGENTS.md"
        agents_dst = self._work_dir / "AGENTS.md"
        if agents_src.exists() and not agents_dst.exists():
            os.symlink(agents_src, agents_dst)

    def ingest_session(self, session: dict) -> dict:
        """Transcript-based ingest: pass session to flush.py (no compile)."""
        # Build transcript
        conv_text = f"[Session: {session.get('timestamp', 'unknown')}]\n"
        for turn in session["conversation"]:
            role = "User" if turn['role'] == 'user' else "Assistant"
            conv_text += f"{role}: {turn['content']}\n"

        self._session_count += 1
        ts = session.get('timestamp', 'unknown')
        print(f"      [karpathy] session {self._session_count} ({ts}) — flush start")

        # Write transcript to temp file (flush.py reads from file)
        context_file = self._work_dir / f"context_{self._session_count}.md"
        context_file.write_text(conv_text, encoding="utf-8")

        # Call flush.py's run_flush directly
        from scripts.flush import run_flush
        from scripts.config import DAILY_DIR

        # Temporarily override DAILY_DIR
        import scripts.config as cfg
        orig_daily = cfg.DAILY_DIR
        cfg.DAILY_DIR = self._work_dir / "daily"

        flush_kwargs = _supported_kwargs(run_flush, model=self._internal_model)
        try:
            result = asyncio.get_event_loop().run_until_complete(run_flush(conv_text, **flush_kwargs))
        except RuntimeError:
            # No event loop running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_flush(conv_text, **flush_kwargs))

        cfg.DAILY_DIR = orig_daily

        flush_status = "FLUSH_OK" if (not result or result.strip() == "FLUSH_OK") else f"saved ({len(result)} chars)"
        print(f"      [karpathy] session {self._session_count} — flush done: {flush_status}")

        # Append result to daily log (use session timestamp as date)
        if result and result.strip() != "FLUSH_OK":
            ts_str = session.get('timestamp', '')
            # Extract date from "2023/03/01 (Wed) 09:00" format
            date_str = ts_str.split('(')[0].strip().replace('/', '-') if ts_str else ""
            if not date_str:
                from datetime import datetime
                date_str = datetime.now().strftime("%Y-%m-%d")
            daily_file = self._work_dir / "daily" / f"{date_str}.md"
            with open(daily_file, "a", encoding="utf-8") as f:
                f.write(f"\n---\n## Session {self._session_count} [{ts_str}]\n{result}\n")

        # Clean up context file
        context_file.unlink(missing_ok=True)

        # Mark that new content needs compilation (deferred to retrieve time)
        if result and result.strip() != "FLUSH_OK":
            self._needs_compile = True

        return {"flush_result": (result or "")[:200], "session": self._session_count}

    def _compile(self):
        """Compile daily logs into knowledge articles."""
        import config as cfg

        # Save originals
        orig = {}
        for attr in ['DAILY_DIR', 'KNOWLEDGE_DIR', 'CONCEPTS_DIR', 'CONNECTIONS_DIR', 'QA_DIR', 'INDEX_FILE', 'LOG_FILE', 'ROOT_DIR', 'STATE_FILE']:
            if hasattr(cfg, attr):
                orig[attr] = getattr(cfg, attr)

        # Override to our workspace (STATE_FILE isolation prevents race conditions
        # across worker processes that all read/write .deps/.../scripts/state.json)
        cfg.ROOT_DIR = self._work_dir
        cfg.DAILY_DIR = self._work_dir / "daily"
        cfg.KNOWLEDGE_DIR = self._work_dir / "knowledge"
        cfg.CONCEPTS_DIR = self._work_dir / "knowledge" / "concepts"
        cfg.CONNECTIONS_DIR = self._work_dir / "knowledge" / "connections"
        cfg.QA_DIR = self._work_dir / "knowledge" / "qa"
        cfg.INDEX_FILE = self._work_dir / "knowledge" / "index.md"
        cfg.LOG_FILE = self._work_dir / "knowledge" / "log.md"
        cfg.STATE_FILE = self._work_dir / "state.json"

        # Force reload utils + compile so they pick up the overridden config paths.
        # Both use "from config import X" which copies values at import time;
        # reloading forces them to re-read the now-overridden cfg attributes.
        # Also override compile.ROOT_DIR (hardcoded via Path(__file__)) so the
        # Claude Agent SDK's cwd points to our workspace, not the original repo.
        import utils as utils_mod
        importlib.reload(utils_mod)
        import compile as compile_mod
        importlib.reload(compile_mod)
        compile_mod.ROOT_DIR = self._work_dir
        from compile import compile_daily_log
        from utils import file_hash

        try:
            daily_files = sorted((self._work_dir / "daily").glob("*.md"))
            skipped = 0
            compiled = 0
            for daily_file in daily_files:
                # Skip daily files that are already compiled AND unchanged.
                # self._compile_state accumulates {filename: {"hash": ...}} from
                # previous compile_daily_log calls in this episode, so the 2nd
                # compile (finalize_ingest) only re-processes new/modified logs.
                prev = self._compile_state.get("ingested", {}).get(daily_file.name, {})
                if prev.get("hash") == file_hash(daily_file):
                    skipped += 1
                    continue
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    cdl_kwargs = _supported_kwargs(compile_daily_log, model=self._internal_model)
                    loop.run_until_complete(compile_daily_log(daily_file, self._compile_state, **cdl_kwargs))
                    compiled += 1
                except Exception as e:
                    print(f"      [karpathy] compile error: {e}")
            print(f"      [karpathy] compile summary: {compiled} compiled, {skipped} skipped (unchanged)")
        finally:
            for k, v in orig.items():
                setattr(cfg, k, v)

    def retrieve(self, question: str) -> str:
        """Karpathy Wiki native retrieval: query.py reads index → selects relevant articles."""
        # Compile if new content has been ingested since last compile
        if self._needs_compile:
            print(f"      [karpathy] retrieve — compiling accumulated daily logs")
            try:
                self._compile()
                n_articles = len(list((self._work_dir / "knowledge" / "concepts").glob("*.md")))
                print(f"      [karpathy] retrieve — compile done ({n_articles} articles)")
            except Exception as e:
                print(f"      [karpathy] retrieve — compile failed: {e}")
            self._needs_compile = False

        print(f"      [karpathy] retrieve — query start: {question[:60]}")
        # Use official query.py (reads index, selects relevant articles, returns answer)
        import config as cfg
        orig = {}
        for attr in ['ROOT_DIR', 'KNOWLEDGE_DIR', 'CONCEPTS_DIR', 'CONNECTIONS_DIR', 'QA_DIR', 'INDEX_FILE', 'STATE_FILE']:
            if hasattr(cfg, attr):
                orig[attr] = getattr(cfg, attr)

        cfg.ROOT_DIR = self._work_dir
        cfg.KNOWLEDGE_DIR = self._work_dir / "knowledge"
        cfg.CONCEPTS_DIR = self._work_dir / "knowledge" / "concepts"
        cfg.CONNECTIONS_DIR = self._work_dir / "knowledge" / "connections"
        cfg.QA_DIR = self._work_dir / "knowledge" / "qa"
        cfg.INDEX_FILE = self._work_dir / "knowledge" / "index.md"
        cfg.STATE_FILE = self._work_dir / "state.json"

        # Reload query + utils so they pick up overridden config paths
        # Also override query.ROOT_DIR (hardcoded via Path(__file__))
        import utils as utils_mod
        importlib.reload(utils_mod)
        import query as query_mod
        importlib.reload(query_mod)
        query_mod.ROOT_DIR = self._work_dir
        from query import run_query

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            rq_kwargs = _supported_kwargs(run_query, model=self._internal_model)
            result = loop.run_until_complete(run_query(question, **rq_kwargs))
            return result.strip() if result else "(empty)"
        finally:
            for k, v in orig.items():
                setattr(cfg, k, v)

    def answer_question(self, question: str, client=None, model=None) -> str:
        """Override: use query.py output directly as the answer.

        Karpathy's query.py already performs retrieval + reasoning in one step.
        Passing its output through UNIFIED_ANSWER_PROMPT strips nuance
        (e.g., uncertainty expressions for Abs, deletion context).
        """
        try:
            from eval.budget_tracker import get_tracker
            _tracker = get_tracker()
        except Exception:
            _tracker = None

        if _tracker is not None:
            _tracker.set_scope("retrieve")
        context = self.retrieve(question)
        self._last_retrieved_context = context

        if _tracker is not None:
            _tracker.set_scope("ingest")

        return context  # query.py output IS the answer

    def finalize_ingest(self):
        """Compile accumulated daily logs into wiki articles."""
        if self._needs_compile:
            print(f"      [karpathy] finalize_ingest — compiling accumulated daily logs")
            self._compile()
            n_articles = len(list((self._work_dir / "knowledge" / "concepts").glob("*.md")))
            print(f"      [karpathy] finalize_ingest — compile done ({n_articles} articles)")
            self._needs_compile = False

    def get_memory_snapshot(self) -> dict:
        """Return full knowledge base content (all articles + daily logs)."""
        parts = []
        knowledge_dir = self._work_dir / "knowledge"
        index_file = knowledge_dir / "index.md"
        if index_file.exists():
            parts.append(f"INDEX:\n{index_file.read_text(encoding='utf-8')}")
        for d in ["concepts", "connections", "qa"]:
            for f in sorted((knowledge_dir / d).glob("*.md")):
                parts.append(f"{d.upper()} ({f.stem}):\n{f.read_text(encoding='utf-8')}")
        for f in sorted((self._work_dir / "daily").glob("*.md")):
            parts.append(f"DAILY ({f.stem}):\n{f.read_text(encoding='utf-8')}")
        text = "\n\n".join(parts) if parts else "(empty)"
        return {"type": "karpathy_wiki", "text": text}

    def reset(self):
        """Clear workspace for new episode."""
        import shutil
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._session_count = 0
        self._last_retrieved_context = ""
        self._needs_compile = False
        self._compile_state = {"ingested": {}}
        self._init_workspace()
