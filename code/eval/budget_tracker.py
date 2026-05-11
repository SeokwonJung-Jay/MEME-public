"""
Budget tracker — counts LLM calls and tokens across all memory systems.

Patches Anthropic Python SDK (sync + async) and Claude Agent SDK once at module
load so every internal LLM call (Mem0 extraction, Graphiti edge extraction,
MD-flat tool loop, Karpathy flush/compile/query) increments a counter.

Three scopes are tracked simultaneously:
  - "ingest":   memory-system calls during ingest_session(...)
  - "retrieve": memory-system calls during retrieve(question)
  - "answer":   the unified answer LLM call (UNIFIED_ANSWER_PROMPT via Sonnet)

The orchestrator (run_agent.py) switches to "ingest" around feed_sessions; the
base.answer_question switches to "retrieve" around self.retrieve(...) and to
"answer" around the final unified-prompt call.

Limitation: counters are process-global. Run with workers=1 (or subprocess
isolation) per memory system to attribute usage to a single episode at a time.
"""
import threading

_LOCK = threading.Lock()


class _ScopeCounter:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self.output_tokens = 0

    def log(self, in_t, cache_create, cache_read, out_t):
        self.calls += 1
        self.input_tokens += int(in_t or 0)
        self.cache_creation_tokens += int(cache_create or 0)
        self.cache_read_tokens += int(cache_read or 0)
        self.output_tokens += int(out_t or 0)

    def reset(self):
        self.calls = 0
        self.input_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self.output_tokens = 0

    def snapshot(self):
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "output_tokens": self.output_tokens,
        }


SCOPES = ("ingest", "retrieve", "answer")


class BudgetTracker:
    def __init__(self):
        self.counters = {s: _ScopeCounter() for s in SCOPES}
        self._scope = "ingest"

    def set_scope(self, scope: str):
        assert scope in SCOPES, f"Unknown scope: {scope}"
        with _LOCK:
            self._scope = scope

    def log(self, in_t: int, cache_create: int, cache_read: int, out_t: int):
        with _LOCK:
            self.counters[self._scope].log(in_t, cache_create, cache_read, out_t)

    def reset(self):
        with _LOCK:
            for c in self.counters.values():
                c.reset()
            self._scope = "ingest"

    def snapshot(self) -> dict:
        with _LOCK:
            return {s: self.counters[s].snapshot() for s in SCOPES}


_tracker = BudgetTracker()


def get_tracker() -> BudgetTracker:
    return _tracker


_PATCHED = False


def install_patches():
    """Install patches on Anthropic SDK (sync + async) + Claude Agent SDK. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    def _extract_usage(usage):
        """Extract (input, cache_creation, cache_read, output) from usage object."""
        if isinstance(usage, dict):
            g = usage.get
        else:
            g = lambda k, d=0: getattr(usage, k, d) or 0
        return (
            g("input_tokens", 0) or 0,
            g("cache_creation_input_tokens", 0) or 0,
            g("cache_read_input_tokens", 0) or 0,
            g("output_tokens", 0) or 0,
        )

    # 1a. Sync Anthropic Messages.create — MD-flat (via AnthropicAsOpenAI adapter), Mem0
    try:
        from anthropic.resources.messages.messages import Messages
        _orig_create = Messages.create

        def _patched_create(self, **kwargs):
            r = _orig_create(self, **kwargs)
            usage = getattr(r, "usage", None)
            if usage is not None:
                _tracker.log(*_extract_usage(usage))
            return r

        Messages.create = _patched_create
    except Exception as e:
        print(f"  [budget_tracker] Anthropic sync patch skipped: {e}")

    # 1b. Async Anthropic Messages.create — Graphiti
    try:
        from anthropic.resources.messages.messages import AsyncMessages
        _orig_acreate = AsyncMessages.create

        async def _patched_acreate(self, **kwargs):
            r = await _orig_acreate(self, **kwargs)
            usage = getattr(r, "usage", None)
            if usage is not None:
                _tracker.log(*_extract_usage(usage))
            return r

        AsyncMessages.create = _patched_acreate
    except Exception as e:
        print(f"  [budget_tracker] Anthropic async patch skipped: {e}")

    # 2. Sync OpenAI ChatCompletions.create — Mem0 native (gpt-5-mini)
    try:
        from openai.resources.chat.completions import Completions as OAICompletions
        _orig_oai_create = OAICompletions.create

        def _patched_oai_create(self, **kwargs):
            r = _orig_oai_create(self, **kwargs)
            usage = getattr(r, "usage", None)
            if usage is not None:
                _tracker.log(
                    getattr(usage, "prompt_tokens", 0) or 0,
                    0, 0,  # OpenAI doesn't have cache fields
                    getattr(usage, "completion_tokens", 0) or 0,
                )
            return r

        OAICompletions.create = _patched_oai_create
    except Exception as e:
        print(f"  [budget_tracker] OpenAI sync patch skipped: {e}")

    # 2b. Async OpenAI ChatCompletions.create — Graphiti native (gpt-4.1-mini)
    try:
        from openai.resources.chat.completions import AsyncCompletions as OAIAsyncCompletions
        _orig_oai_acreate = OAIAsyncCompletions.create

        async def _patched_oai_acreate(self, **kwargs):
            r = await _orig_oai_acreate(self, **kwargs)
            usage = getattr(r, "usage", None)
            if usage is not None:
                _tracker.log(
                    getattr(usage, "prompt_tokens", 0) or 0,
                    0, 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                )
            return r

        OAIAsyncCompletions.create = _patched_oai_acreate
    except Exception as e:
        print(f"  [budget_tracker] OpenAI async patch skipped: {e}")

    # 2c. Async OpenAI Responses.parse — Graphiti structured completions
    try:
        from openai.resources.responses.responses import AsyncResponses as OAIAsyncResponses
        _orig_oai_responses_parse = OAIAsyncResponses.parse

        async def _patched_oai_responses_parse(self, **kwargs):
            r = await _orig_oai_responses_parse(self, **kwargs)
            usage = getattr(r, "usage", None)
            if usage is not None:
                _tracker.log(
                    getattr(usage, "input_tokens", 0) or 0,
                    0, 0,
                    getattr(usage, "output_tokens", 0) or 0,
                )
            return r

        OAIAsyncResponses.parse = _patched_oai_responses_parse
    except Exception as e:
        print(f"  [budget_tracker] OpenAI async responses.parse patch skipped: {e}")

    # 2d. Async OpenAI Responses.create — fallback for other response API calls
    try:
        from openai.resources.responses.responses import AsyncResponses as OAIAsyncResponses2
        _orig_oai_responses_create = OAIAsyncResponses2.create

        async def _patched_oai_responses_create(self, **kwargs):
            r = await _orig_oai_responses_create(self, **kwargs)
            usage = getattr(r, "usage", None)
            if usage is not None:
                _tracker.log(
                    getattr(usage, "input_tokens", 0) or 0,
                    0, 0,
                    getattr(usage, "output_tokens", 0) or 0,
                )
            return r

        OAIAsyncResponses2.create = _patched_oai_responses_create
    except Exception as e:
        print(f"  [budget_tracker] OpenAI async responses.create patch skipped: {e}")

    # 3. Claude Agent SDK query — Karpathy (heavy prompt caching via Claude Code preset)
    try:
        import claude_agent_sdk

        _orig_query = claude_agent_sdk.query

        async def _patched_query(*args, **kwargs):
            async for msg in _orig_query(*args, **kwargs):
                usage = getattr(msg, "usage", None)
                if usage:
                    _tracker.log(*_extract_usage(usage))
                yield msg

        claude_agent_sdk.query = _patched_query
    except Exception as e:
        print(f"  [budget_tracker] Claude Agent SDK patch skipped: {e}")
