"""
GraphitiMemory — Memory system using Graphiti/Zep (https://github.com/getzep/graphiti).

Architecture:
  - Ingest: session transcript → add_episode() → Graphiti internal LLM extracts entities/relations → Neo4j graph
  - Retrieve: question → client.search() → returns relevant edges/facts
  - Answer: unified LLM (from base) uses retrieved context

Requires: Neo4j running on localhost:7687, Python 3.10+, graphiti-core pip package.
Run this agent via the graphiti venv: source /tmp/graphiti_env/bin/activate
"""

import os
import asyncio
from datetime import datetime, timezone
from agents.base import BaseMemorySystem


# Process-global current session info; graphiti's LLM patch reads this so
# error log lines tag the right session (filler vs evidence) that triggered
# the retry. Set in ingest_session().
_CURRENT_SESSION = None  # tuple (session_id, type) or None


def _log_glm_event(self_ref, attempt, kind, exc):
    """Per-worker GLM retry log. Each worker writes to its own pid-tagged
    file so output isn't interleaved or buffered-lost via tee.
    `kind` ∈ {'retry', 'recovered', 'gave_up'}."""
    try:
        ep = getattr(self_ref, '_group_id', None) or 'unknown'
        sess = _CURRENT_SESSION or ('?', '?')
        msg = f'{type(exc).__name__}: {str(exc)[:200]}' if exc else ''
        line = (f'[pid={os.getpid()} ep={ep} session={sess[0]} type={sess[1]} '
                f'attempt={attempt+1}/3 {kind}] {msg}\n')
        path = os.environ.get('GLM_RETRY_LOG', f'/tmp/glm_retry.pid{os.getpid()}.log')
        with open(path, 'a') as f:
            f.write(line)
    except Exception:
        pass


class GraphitiMemory(BaseMemorySystem):
    """Graphiti-based temporal knowledge graph memory system.

    Transcript-based ingest: session text → Graphiti API directly.
    """

    def __init__(self, model="claude-sonnet-4-20250514", api_key=None,
                 neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="mempass123",
                 internal_model=None, **kwargs):
        from graphiti_core import Graphiti

        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._group_id = None
        self._internal_model = internal_model

        if internal_model:
            if internal_model.startswith("claude"):
                from graphiti_core.llm_client.anthropic_client import AnthropicClient
                from graphiti_core.llm_client.config import LLMConfig
                # Read ANTHROPIC_API_KEY directly: the answering-LLM api_key
                # may be the OpenAI key when --model is non-Claude, which
                # would 401 the Anthropic internal client.
                anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
                llm_client = AnthropicClient(LLMConfig(model=internal_model, api_key=anthropic_key))
                self.client = Graphiti(neo4j_uri, neo4j_user, neo4j_password, llm_client=llm_client)
            else:
                from graphiti_core.llm_client.openai_client import OpenAIClient
                from graphiti_core.llm_client.config import LLMConfig
                from agents._model_utils import is_openrouter_model, OPENROUTER_BASE_URL
                # OpenRouter models (e.g., z-ai/glm-5.1) need separate key + base_url;
                # plain OpenAI models keep the default endpoint unchanged.
                _is_or = is_openrouter_model(internal_model)
                if _is_or:
                    openai_key = os.environ.get("OPENROUTER_API_KEY")
                    base_url = OPENROUTER_BASE_URL
                else:
                    openai_key = os.environ.get("OPENAI_API_KEY")
                    base_url = None
                llm_client = OpenAIClient(LLMConfig(model=internal_model, api_key=openai_key, base_url=base_url))

                # OpenRouter quirks for reasoning/structured models:
                # 1. Wraps output as `reasoning_text` blocks → inject
                #    `reasoning.exclude=True` so the reasoning wrapper is dropped.
                # 2. Strips `strict: true` from structured schemas unless the
                #    `structured-outputs-2025-11-13` header is sent → include it
                #    so OpenRouter enforces exact field names via json_schema.
                # 3. Some OpenRouter models (GLM-5 family) wrap JSON in
                #    markdown code fences (```json ... ```), which breaks
                #    Pydantic parsing. Patch OpenAI SDK's parse_text to strip
                #    fences before validation.
                if _is_or:
                    _orig_parse = llm_client.client.responses.parse
                    self_ref = self  # capture for ep-id logging in closure
                    async def _parse_or(**kwargs):
                        eb = kwargs.get('extra_body') or {}
                        eb.setdefault('reasoning', {})
                        # GLM-5.1 via OpenRouter: reasoning ON returns
                        # content=None / output_parsed=None on complex
                        # extraction prompts. Disabling reasoning is the only
                        # configuration that produces extractable content.
                        # Asymmetry vs GPT-5 documented in paper.
                        eb['reasoning']['enabled'] = False
                        eb.setdefault('provider', {})
                        eb['provider']['ignore'] = ['novita']
                        kwargs['extra_body'] = eb
                        eh = kwargs.get('extra_headers') or {}
                        eh.setdefault('structured-outputs-2025-11-13', 'true')
                        kwargs['extra_headers'] = eh
                        # Retry-on-failure: up to 3 attempts to absorb GLM's
                        # rare bad outputs (truncation, reasoning-only, schema
                        # validation). Same policy as mem0 — we measure memory
                        # systems, not internal-LLM reliability.
                        last_exc = None
                        for _attempt in range(3):
                            try:
                                resp = await _orig_parse(**kwargs)
                                if getattr(resp, 'output_parsed', None) is not None:
                                    if _attempt > 0:
                                        _log_glm_event(self_ref, _attempt, 'recovered', None)
                                    return resp
                                last_exc = ValueError('output_parsed is None')
                            except Exception as e:
                                last_exc = e
                            _log_glm_event(self_ref, _attempt, 'retry', last_exc)
                        _log_glm_event(self_ref, 2, 'gave_up', last_exc)
                        raise last_exc
                    llm_client.client.responses.parse = _parse_or

                    # Strip markdown code fences before Pydantic parses the
                    # response text. Applied once at module level; benign for
                    # clean-JSON responses (no-op when no fence present).
                    import openai.lib._parsing._responses as _resp_parsing
                    if not getattr(_resp_parsing, '_meme_fence_patched', False):
                        import json as _json
                        _orig_parse_text = _resp_parsing.parse_text
                        def _patched_parse_text(text, text_format):
                            if isinstance(text, str):
                                s = text.strip()
                                # (a) strip markdown code fences
                                if s.startswith('```'):
                                    lines = s.split('\n')
                                    lines = lines[1:]
                                    if lines and lines[-1].strip().startswith('```'):
                                        lines = lines[:-1]
                                    text = '\n'.join(lines)
                                    s = text.strip()
                                # (b) wrap raw array if schema expects a single
                                # array-valued field (GLM-5 sometimes drops the
                                # outer object).
                                if s.startswith('['):
                                    try:
                                        arr = _json.loads(s)
                                    except Exception:
                                        arr = None
                                    if isinstance(arr, list) and hasattr(text_format, 'model_json_schema'):
                                        try:
                                            schema = text_format.model_json_schema()
                                            props = schema.get('properties', {})
                                            if len(props) == 1:
                                                fname = next(iter(props))
                                                if props[fname].get('type') == 'array':
                                                    text = _json.dumps({fname: arr})
                                        except Exception:
                                            pass
                            return _orig_parse_text(text, text_format)
                        _resp_parsing.parse_text = _patched_parse_text
                        _resp_parsing._meme_fence_patched = True
                self.client = Graphiti(neo4j_uri, neo4j_user, neo4j_password, llm_client=llm_client)
        else:
            # Native default: Graphiti's built-in LLM (gpt-4.1-mini)
            self.client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

        # Build indices (async) — skip if pre-built via start_neo4j_cluster.sh
        if not os.environ.get("NEO4J_SKIP_INDEX_BUILD"):
            asyncio.get_event_loop().run_until_complete(
                self.client.build_indices_and_constraints()
            )

        self._last_retrieved_context = ""
        self._episode_count = 0

    def _set_group_id(self, group_id):
        """Set group_id for namespace isolation (per-episode)."""
        self._group_id = str(group_id)

    def ingest_session(self, session: dict) -> dict:
        """Transcript-based ingest: pass session to Graphiti add_episode."""
        # Track current session so retry-log entries can be attributed to
        # the specific session (filler vs evidence) that triggered them.
        global _CURRENT_SESSION
        _CURRENT_SESSION = (session.get('session_id', '?'), session.get('type', '?'))

        conv_text = f"[Session: {session.get('timestamp', 'unknown')}]\n"
        for turn in session["conversation"]:
            role = "User" if turn['role'] == 'user' else "Assistant"
            conv_text += f"{role}: {turn['content']}\n"

        self._episode_count += 1

        result = asyncio.get_event_loop().run_until_complete(
            self.client.add_episode(
                name=f"session_{self._episode_count}",
                episode_body=conv_text,
                source_description="conversation",
                reference_time=datetime.now(timezone.utc),
                group_id=self._group_id or "default",
            )
        )
        return {"graphiti_result": "episode_added"}

    def retrieve(self, question: str) -> str:
        """Graphiti native retrieval: hybrid search (semantic + BM25 + graph traversal)."""
        results = asyncio.get_event_loop().run_until_complete(
            self.client.search(
                question,
                group_ids=[self._group_id] if self._group_id else None,
                num_results=10,
            )
        )

        if not results:
            return "(no relevant facts)"

        lines = []
        for edge in results:
            fact = getattr(edge, 'fact', str(edge))
            name = getattr(edge, 'name', '')
            lines.append(f"[{name}] {fact}")

        return "\n".join(lines)

    def get_memory_snapshot(self) -> dict:
        """Return all graph edges for this group via direct API (not search)."""
        try:
            group_ids = [self._group_id] if self._group_id else []
            entity_edges = asyncio.get_event_loop().run_until_complete(
                self.client.edges.entity.get_by_group_ids(
                    group_ids=group_ids,
                    limit=None,
                )
            )
            lines = []
            for edge in entity_edges:
                fact = getattr(edge, 'fact', str(edge))
                lines.append(fact)
            text = "\n".join(lines)
        except Exception as e:
            text = f"(unable to retrieve snapshot: {e})"

        return {"type": "graphiti", "text": text or "(empty)"}

    def reset(self):
        """Reset for new episode. Cleans up ALL data from this Neo4j instance."""
        try:
            import time as _time
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_password))
            with driver.session() as session:
                before = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                session.run("MATCH (n) DETACH DELETE n")
                after = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            _time.sleep(1)
            with driver.session() as session:
                after2 = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                print(f"      [graphiti] reset on {self._neo4j_uri}: {before} → {after} → (1s later) {after2} nodes (group={self._group_id})", flush=True)
            driver.close()
        except Exception as e:
            print(f"      [graphiti] cleanup failed: {e}", flush=True)
        self._group_id = None
        self._episode_count = 0
        self._last_retrieved_context = ""

    def close(self):
        """Close Neo4j connection."""
        asyncio.get_event_loop().run_until_complete(self.client.close())
