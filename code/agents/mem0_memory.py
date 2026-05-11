"""
Mem0Memory — Memory system using Mem0 (https://github.com/mem0ai/mem0).

Architecture:
  - Ingest: session transcript → m.add() → Mem0 internal LLM extracts facts → vector+graph storage
  - Retrieve: question → m.search() → returns relevant memories
  - Answer: unified LLM (from base) uses retrieved context

Mem0 handles its own fact extraction, self-edit (updates on conflict), and semantic search.
"""

import os
from agents.base import BaseMemorySystem


# Process-global current ep id + session info; mem0's LLM patch reads these
# so error log lines tag the right episode/session even though the patch
# sits inside OpenAILLM (which has no episode awareness).
_CURRENT_EP_ID = None
_CURRENT_SESSION = None  # tuple (session_id, type) or None


def _log_glm_event(_unused_self, attempt, kind, exc):
    """Per-worker GLM retry log. Each worker writes to its own pid-tagged
    file so output isn't interleaved or buffered-lost via tee.
    `kind` ∈ {'retry', 'recovered', 'gave_up'}."""
    try:
        ep = _CURRENT_EP_ID or 'unknown'
        sess = _CURRENT_SESSION or ('?', '?')
        msg = f'{type(exc).__name__}: {str(exc)[:200]}' if exc else ''
        line = (f'[pid={os.getpid()} ep={ep} session={sess[0]} type={sess[1]} '
                f'attempt={attempt+1}/3 {kind}] {msg}\n')
        path = os.environ.get('GLM_RETRY_LOG', f'/tmp/glm_retry.pid{os.getpid()}.log')
        with open(path, 'a') as f:
            f.write(line)
    except Exception:
        pass


class Mem0Memory(BaseMemorySystem):
    """Mem0-based memory system.

    Transcript-based ingest: session text → Mem0 API directly (no agent LLM for ingest).
    """

    def __init__(self, model="claude-sonnet-4-20250514", api_key=None, internal_model=None, top_k=None, **kwargs):
        # top_k=None preserves Mem0's library default (top-20 in our pinned
        # version, see app:systems). Pass an explicit value only for the
        # tab:topk-sweep ablation.
        self._top_k = top_k
        from mem0 import Memory
        import uuid

        # OpenRouter reasoning models (GLM-5/5.1) sometimes return content=None
        # because reasoning_text consumes the response. Two layered patches:
        #   (1) inject extra_body={"reasoning":{"exclude":True}} on every
        #       OpenAI client call so OpenRouter strips the reasoning wrapper.
        #   (2) make remove_code_blocks tolerate None inputs so a still-empty
        #       response doesn't crash the extraction pipeline.
        from agents._model_utils import is_openrouter_model
        if internal_model and is_openrouter_model(internal_model):
            import mem0.memory.utils as _mem0_utils
            if not getattr(_mem0_utils, '_meme_or_patched', False):
                _orig_rcb = _mem0_utils.remove_code_blocks
                def _safe_rcb(content):
                    if content is None:
                        return ""
                    return _orig_rcb(content)
                _mem0_utils.remove_code_blocks = _safe_rcb
                _mem0_utils._meme_or_patched = True
                # Also patch the symbol re-imported into main.py
                try:
                    import mem0.memory.main as _mem0_main
                    _mem0_main.remove_code_blocks = _safe_rcb
                except Exception:
                    pass

            import mem0.llms.openai as _mem0_openai_mod
            if not getattr(_mem0_openai_mod, '_meme_or_extra_body_patched', False):
                _orig_init = _mem0_openai_mod.OpenAILLM.__init__
                def _patched_init(self_llm, config=None):
                    _orig_init(self_llm, config)
                    if os.environ.get("OPENROUTER_API_KEY"):
                        _orig_create = self_llm.client.chat.completions.create
                        def _create_with_reasoning_exclude(**kw):
                            eb = kw.get('extra_body') or {}
                            eb.setdefault('reasoning', {})
                            # GLM-5.1 via OpenRouter: reasoning ON (any
                            # effort, any provider) returns content=None on
                            # complex extraction prompts. Disabling reasoning
                            # is the only configuration that produces
                            # extractable content. Asymmetry vs GPT-5
                            # (reasoning_effort=minimal) documented in paper.
                            eb['reasoning']['enabled'] = False
                            # Novita is excluded as additional safety —
                            # provided the cleanest behavior in our tests.
                            eb.setdefault('provider', {})
                            eb['provider']['ignore'] = ['novita']
                            kw['extra_body'] = eb
                            # When response_format demands JSON, retry on
                            # malformed content. We're evaluating the memory
                            # system, not GLM-5.1's output reliability.
                            import json as _json
                            from mem0.memory.utils import extract_json as _extract_json
                            wants_json = (kw.get('response_format') or {}).get('type') == 'json_object'
                            last_reason = None
                            for _attempt in range(3):
                                resp = _orig_create(**kw)
                                if not wants_json:
                                    return resp
                                try:
                                    content = resp.choices[0].message.content
                                except Exception:
                                    return resp
                                if content is None:
                                    last_reason = ValueError('content=None')
                                    _log_glm_event(None, _attempt, 'retry', last_reason)
                                    continue
                                try:
                                    _json.loads(content, strict=False)
                                    if _attempt > 0:
                                        _log_glm_event(None, _attempt, 'recovered', None)
                                    return resp
                                except Exception as e:
                                    last_reason = e
                                try:
                                    _json.loads(_extract_json(content), strict=False)
                                    if _attempt > 0:
                                        _log_glm_event(None, _attempt, 'recovered_via_extract', None)
                                    return resp
                                except Exception as e:
                                    last_reason = e
                                _log_glm_event(None, _attempt, 'retry', last_reason)
                            _log_glm_event(None, 2, 'gave_up', last_reason)
                            return resp
                        self_llm.client.chat.completions.create = _create_with_reasoning_exclude
                _mem0_openai_mod.OpenAILLM.__init__ = _patched_init
                _mem0_openai_mod._meme_or_extra_body_patched = True

        # Use Qdrant Docker server (not qdrant-local) to avoid SQLite thread errors.
        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))
        collection_name = f"mem0_{uuid.uuid4().hex[:8]}"
        self._collection_name = collection_name

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
        }

        if internal_model:
            provider = "anthropic" if internal_model.startswith("claude") else "openai"
            key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            llm_conf = {
                "model": internal_model,
                "api_key": api_key or os.environ.get(key_env),
            }
            # Reasoning models (gpt-5 family) require reasoning_effort and
            # reject temperature/max_tokens. Mem0 auto-strips unsupported params.
            from agents._model_utils import is_reasoning_model, DEFAULT_REASONING_EFFORT
            if provider == "openai" and is_reasoning_model(internal_model):
                llm_conf["reasoning_effort"] = DEFAULT_REASONING_EFFORT
            config["llm"] = {"provider": provider, "config": llm_conf}
        # else: no llm config → Mem0 uses its native default (gpt-5-mini)

        self.memory = Memory.from_config(config)
        self._user_id = None
        self._last_retrieved_context = ""
        self._internal_model = internal_model

    def _set_user_id(self, user_id):
        """Set user_id for namespace isolation (per-episode)."""
        self._user_id = str(user_id)
        # Mirror to module-global so the OpenAILLM patch (which has no
        # episode context) can tag retry-log lines with this ep id.
        global _CURRENT_EP_ID
        _CURRENT_EP_ID = self._user_id

    def ingest_session(self, session: dict) -> dict:
        """Transcript-based ingest: pass session conversation to Mem0.

        Mem0 embeds the transcript with text-embedding-3-small (8192-token
        input limit). Long sessions (e.g., SW filler with huge code-block
        turns) would raise BadRequestError. We pre-split using the same
        criteria as DenseMemory: turn-boundary chunks at 4096 tokens, with a
        sliding-window fallback (max 8000 tok, 400 overlap) for any single
        turn that exceeds the chunk budget.
        """
        # Track current session so retry-log entries can be attributed to
        # the specific session (filler vs evidence) that triggered them.
        global _CURRENT_SESSION
        _CURRENT_SESSION = (session.get('session_id', '?'), session.get('type', '?'))

        from agents._chunking import session_to_chunks
        from agents.dense_memory import DenseMemory

        chunks = session_to_chunks(session, max_tokens=4096)
        safe_chunks = [w for c in chunks for w in DenseMemory._split_for_embedding(c)]

        results = []
        for chunk_text in safe_chunks:
            r = self.memory.add(chunk_text, user_id=self._user_id)
            results.append(r)
        return {"mem0_result": results, "n_chunks": len(safe_chunks)}

    def retrieve(self, question: str) -> str:
        """Mem0 native retrieval: semantic search."""
        search_kwargs = {"query": question, "filters": {"user_id": self._user_id}}
        if self._top_k is not None:
            search_kwargs["limit"] = self._top_k
        results = self.memory.search(**search_kwargs)

        # Format results as context string
        memories = results.get("results", [])
        if not memories:
            return "(no relevant facts)"

        lines = []
        for mem in memories:
            text = mem.get("memory", "")
            score = mem.get("score", 0)
            lines.append(f"[score={score:.2f}] {text}")

        return "\n".join(lines)

    def get_memory_snapshot(self) -> dict:
        """Return all Mem0 memories for this user."""
        all_mems = self.memory.get_all(filters={"user_id": self._user_id}, top_k=10000)
        memories = all_mems.get("results", [])

        text = "\n".join(
            f"[{m.get('id', '?')[:8]}] {m.get('memory', '')}"
            for m in memories
        )
        return {"type": "mem0", "text": text or "(empty)", "raw": memories}

    def reset(self):
        """Reset for new episode. Cleans up previous collection to prevent memory buildup."""
        from mem0 import Memory
        import uuid

        # Delete previous collection from Qdrant
        if hasattr(self, '_collection_name') and self._collection_name:
            try:
                from qdrant_client import QdrantClient
                qc = QdrantClient(host=os.environ.get("QDRANT_HOST", "localhost"),
                                  port=int(os.environ.get("QDRANT_PORT", "6333")))
                qc.delete_collection(self._collection_name)
            except Exception:
                pass

        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))
        self._collection_name = f"mem0_{uuid.uuid4().hex[:8]}"

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": qdrant_host,
                    "port": qdrant_port,
                    "collection_name": self._collection_name,
                    "embedding_model_dims": 1536,
                },
            },
        }
        if self._internal_model:
            provider = "anthropic" if self._internal_model.startswith("claude") else "openai"
            key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            llm_conf = {
                "model": self._internal_model,
                "api_key": os.environ.get(key_env),
            }
            from agents._model_utils import is_reasoning_model, DEFAULT_REASONING_EFFORT
            if provider == "openai" and is_reasoning_model(self._internal_model):
                llm_conf["reasoning_effort"] = DEFAULT_REASONING_EFFORT
            config["llm"] = {"provider": provider, "config": llm_conf}
        self.memory = Memory.from_config(config)
        self._last_retrieved_context = ""
