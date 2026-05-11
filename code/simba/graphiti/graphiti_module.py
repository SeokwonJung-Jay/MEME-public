"""DSPy Module wrapping Graphiti episode pipeline.

SIMBA optimizes 3 Signatures (mapped onto graphiti_core prompt_library entries):
  - ExtractNodesSig → extract_nodes.extract_message  (entity extraction per message)
  - ExtractEdgesSig → extract_edges.edge             (fact/relation extraction)
  - DedupeNodesSig  → dedupe_nodes.nodes             (batched entity dedup)

No modifications to graphiti_core. `graphiti_prompts_override.py` monkey-patches
the `.func` on `prompt_library.*.*` entries at import time and reads the
thread-local "current instructions" dict that THIS module updates before each
Graphiti call.

Parallelization: each SIMBA worker thread gets its own Neo4j port (from a pool
started by eval/start_neo4j_cluster.sh) AND its own thread-local prompt state.
Run SIMBA with num_threads == num Neo4j instances started.
"""
import asyncio
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import List, Dict

import dspy

from graphiti_prompts_override import (
    install_overrides,
    set_current_instructions,
    DEFAULT_EXTRACT_MESSAGE_INSTRUCTIONS,
    DEFAULT_EDGE_INSTRUCTIONS,
    DEFAULT_DEDUPE_NODES_INSTRUCTIONS,
)

install_overrides()


# ============================================================
# DSPy Signatures (instructions = graphiti_core defaults at baseline)
# ============================================================

class ExtractNodesSig(dspy.Signature):
    __doc__ = DEFAULT_EXTRACT_MESSAGE_INSTRUCTIONS

    conversation_message = dspy.InputField(desc="Current conversation message plus entity types and previous messages as context")
    entities = dspy.OutputField(desc="Extracted entity nodes")


class ExtractEdgesSig(dspy.Signature):
    __doc__ = DEFAULT_EDGE_INSTRUCTIONS

    message_and_entities = dspy.InputField(desc="Current message, previous messages, list of entities, and reference time")
    edges = dspy.OutputField(desc="Fact triples (source, target, relation, valid_at, invalid_at)")


class DedupeNodesSig(dspy.Signature):
    __doc__ = DEFAULT_DEDUPE_NODES_INSTRUCTIONS

    extracted_and_existing = dspy.InputField(desc="Newly-extracted entities plus existing entities from the graph")
    resolutions = dspy.OutputField(desc="Per-entity duplicate resolution")


# ============================================================
# Per-thread Graphiti client + Neo4j port assignment
# ============================================================

_tl = threading.local()
_port_assign_lock = threading.Lock()
_assigned_ports: list[int] = []  # ports already handed out to threads


def _assign_port_for_thread(base_port: int, num_ports: int) -> int:
    """Return a Neo4j bolt port unique to the current thread. Sticky per thread."""
    if hasattr(_tl, "neo4j_port"):
        return _tl.neo4j_port
    with _port_assign_lock:
        # Pick next unused port; if all used (threads > ports), wrap around.
        candidate_order = [base_port + i for i in range(num_ports)]
        for p in candidate_order:
            if p not in _assigned_ports:
                _assigned_ports.append(p)
                _tl.neo4j_port = p
                return p
        # Fallback: wrap (shouldn't happen if num_threads == num_ports)
        _tl.neo4j_port = base_port + (len(_assigned_ports) % num_ports)
        _assigned_ports.append(_tl.neo4j_port)
        return _tl.neo4j_port


def _get_thread_client(neo4j_user: str, neo4j_password: str,
                       internal_model: str, base_port: int, num_ports: int):
    """Return (client, loop, neo4j_uri) for the current thread, lazily creating them."""
    if hasattr(_tl, "client"):
        return _tl.client, _tl.loop, _tl.neo4j_uri

    from graphiti_core import Graphiti
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.llm_client.config import LLMConfig

    port = _assign_port_for_thread(base_port, num_ports)
    uri = f"bolt://localhost:{port}"

    _tl.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_tl.loop)

    openai_key = os.environ.get("OPENAI_API_KEY")
    llm_client = OpenAIClient(LLMConfig(model=internal_model, api_key=openai_key))
    _tl.client = Graphiti(uri, neo4j_user, neo4j_password, llm_client=llm_client)
    _tl.neo4j_uri = uri

    if not os.environ.get("NEO4J_SKIP_INDEX_BUILD"):
        _tl.loop.run_until_complete(_tl.client.build_indices_and_constraints())

    import logging
    logging.info(f"[graphiti thread {threading.current_thread().name}] bound to {uri}")
    return _tl.client, _tl.loop, _tl.neo4j_uri


def _reset_neo4j_group(neo4j_uri: str, neo4j_user: str, neo4j_password: str, group_id: str):
    """Delete all nodes for this group_id. Other groups on the same instance stay intact."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session() as session:
            session.run("MATCH (n {group_id: $gid}) DETACH DELETE n", gid=group_id)
    finally:
        driver.close()


# ============================================================
# DSPy Module
# ============================================================

class GraphitiProgram(dspy.Module):
    """Full Graphiti episode pipeline.

    For each episode:
      1. Copy this program's 3 Signature instructions into thread-local state.
      2. Generate a unique group_id (isolates this episode's subgraph).
      3. Phase 1: add_episode for sessions[0:before_pos], answer `before` questions via search.
      4. Phase 2: add_episode for sessions[before_pos:after_pos], answer `after` questions.
      5. Delete the group from Neo4j.
    """

    def __init__(self,
                 internal_model: str = "gpt-4.1-mini",
                 answer_lm=None,
                 answer_model: str = "claude-sonnet-4-20250514",
                 neo4j_base_port: int = 7687,
                 neo4j_num_ports: int = 1,
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "mempass123"):
        super().__init__()
        self.extract_nodes = dspy.Predict(ExtractNodesSig)
        self.extract_edges = dspy.Predict(ExtractEdgesSig)
        self.dedupe_nodes = dspy.Predict(DedupeNodesSig)
        self.internal_model = internal_model
        self.answer_lm = answer_lm
        self.answer_model = answer_model
        self.neo4j_base_port = neo4j_base_port
        self.neo4j_num_ports = neo4j_num_ports
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

    def _apply_prompts(self):
        set_current_instructions(
            extract_message=self.extract_nodes.signature.instructions,
            edge=self.extract_edges.signature.instructions,
            dedupe_nodes=self.dedupe_nodes.signature.instructions,
        )

    def _ingest_sessions(self, client, loop, sessions, group_id, start_idx=0):
        for i, sess in enumerate(sessions):
            conv_text = f"[Session: {sess.get('timestamp', 'unknown')}]\n"
            for turn in sess["conversation"]:
                role = "User" if turn['role'] == 'user' else "Assistant"
                conv_text += f"{role}: {turn['content']}\n"

            ref_time = datetime.now(timezone.utc)
            ts_str = sess.get('timestamp', '')
            if ts_str:
                parts = ts_str.split(')')
                date_part = parts[0].split('(')[0].strip()
                time_part = parts[1].strip() if len(parts) > 1 else "00:00"
                ref_time = datetime.strptime(f"{date_part} {time_part}",
                                             "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)

            loop.run_until_complete(client.add_episode(
                name=f"session_{start_idx + i + 1}_{group_id}",
                episode_body=conv_text,
                source_description="conversation",
                reference_time=ref_time,
                group_id=group_id,
            ))

    def _search(self, client, loop, question: str, group_id: str) -> str:
        results = loop.run_until_complete(client.search(
            question,
            group_ids=[group_id],
            num_results=10,
        ))
        if not results:
            return "(no relevant facts)"
        lines = []
        for edge in results:
            fact = getattr(edge, 'fact', str(edge))
            name = getattr(edge, 'name', '')
            lines.append(f"[{name}] {fact}")
        return "\n".join(lines)

    def _answer(self, question: str, context: str) -> str:
        answer_instr = ("Answer the user's question based ONLY on the context provided. "
                        "If the information is not in the context, say you don't have that information. "
                        "Answer with ONLY the value. Do not explain or add context.")
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        if self.answer_lm is not None:
            resp = self.answer_lm(
                messages=[
                    {"role": "system", "content": answer_instr},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=500,
            )
            if isinstance(resp, list) and resp:
                resp = resp[0]
            if hasattr(resp, 'choices'):
                return resp.choices[0].message.content or ""
            return str(resp)
        return context

    def _answer_questions(self, client, loop, questions, group_id):
        out = []
        for q in questions:
            facts = self._search(client, loop, q['question'], group_id)
            ans = self._answer(q['question'], facts)
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

        self._apply_prompts()

        client, loop, neo4j_uri = _get_thread_client(
            self.neo4j_user, self.neo4j_password, self.internal_model,
            self.neo4j_base_port, self.neo4j_num_ports,
        )

        group_id = f"simba_{domain}_{episode_id}_{uuid.uuid4().hex[:8]}"
        logging.info(f"[graphiti {domain} ep{episode_id}] START — {neo4j_uri}, group={group_id}, "
                     f"extract_instr_len={len(self.extract_nodes.signature.instructions)}, "
                     f"edge_instr_len={len(self.extract_edges.signature.instructions)}, "
                     f"dedupe_instr_len={len(self.dedupe_nodes.signature.instructions)}")

        try:
            t0 = time.time()
            self._ingest_sessions(client, loop, sessions[:before_pos], group_id, start_idx=0)
            logging.info(f"[graphiti {domain} ep{episode_id}] Phase 1 ingest done in {time.time()-t0:.1f}s")

            before_answers = self._answer_questions(client, loop, before_questions, group_id)

            t0 = time.time()
            self._ingest_sessions(client, loop, sessions[before_pos:after_pos], group_id,
                                  start_idx=before_pos)
            logging.info(f"[graphiti {domain} ep{episode_id}] Phase 2 ingest done in {time.time()-t0:.1f}s")

            after_answers = self._answer_questions(client, loop, after_questions, group_id)

            return dspy.Prediction(
                episode_id=episode_id,
                domain=domain,
                before_answers=before_answers,
                after_answers=after_answers,
            )
        finally:
            _reset_neo4j_group(neo4j_uri, self.neo4j_user, self.neo4j_password, group_id)
