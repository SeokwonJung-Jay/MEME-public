"""
Agent Runner (Orchestrator)
=============================
Runs an agent on assembled episodes and saves outputs for judging.

This file is an orchestrator only — it imports agent types from agents/
and calls the BaseMemorySystem interface. No agent-internal logic lives here.

Usage:
  python3 run_agent.py -d ../data/filler32k_pl --agent-type md_file --model gpt-4.1-mini
"""

import json
import os
import sys
import argparse
import time
from typing import List, Dict

# Add project root to path so we can import agents/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI

from agents.base import BaseMemorySystem
from eval.budget_tracker import install_patches, get_tracker

# Patch SDKs once so every internal LLM call across all systems is counted
install_patches()


# ============================================================
# Agent Factory — the ONLY place with agent-type branching
# ============================================================

def _make_client(model: str, api_key: str = None):
    """Create appropriate client based on model name."""
    if model.startswith("claude"):
        from agents.anthropic_adapter import AnthropicAsOpenAI
        anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        return AnthropicAsOpenAI(api_key=anthropic_key)
    return None  # agents will create their own OpenAI client


def create_agent(agent_type: str, model: str, api_key: str = None,
                 internal_model: str = None, **kwargs) -> BaseMemorySystem:
    """Create an agent instance by type. This is the single dispatch point.

    internal_model: if set, override all systems' internal LLM to this model.
    If not set, each system uses its native default.
    """
    client = _make_client(model, api_key)

    if agent_type == "md_file":
        from agents.md_file import MDFlatMemory
        return MDFlatMemory(model=model, api_key=api_key, client=client,
                           internal_model=internal_model)
    elif agent_type == "mem0":
        from agents.mem0_memory import Mem0Memory
        return Mem0Memory(model=model, api_key=api_key, internal_model=internal_model)
    elif agent_type == "graphiti":
        from agents.graphiti_memory import GraphitiMemory
        return GraphitiMemory(model=model, api_key=api_key, internal_model=internal_model, **kwargs)
    elif agent_type == "karpathy":
        from agents.karpathy_system import KarpathyWikiMemory
        return KarpathyWikiMemory(model=model, api_key=api_key, internal_model=internal_model)
    elif agent_type == "bm25":
        from agents.bm25_memory import BM25Memory
        return BM25Memory(model=model, api_key=api_key, internal_model=internal_model, **kwargs)
    elif agent_type == "dense":
        from agents.dense_memory import DenseMemory
        return DenseMemory(model=model, api_key=api_key, internal_model=internal_model, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ============================================================
# Episode Runner — uses BaseMemorySystem interface only
# ============================================================

def feed_sessions(agent: BaseMemorySystem, sessions: List[Dict],
                  from_index: int, to_index: int) -> List[Dict]:
    """Feed sessions to agent, return ingest logs with per-session memory snapshots."""
    ingest_logs = []
    for i in range(from_index, to_index):
        sess = sessions[i]
        # Mark budget scope as ingest while feeding the session
        get_tracker().set_scope("ingest")
        t0 = time.time()
        result = agent.ingest_session(sess)
        ingest_elapsed = time.time() - t0

        # Capture memory snapshot after each session
        snapshot = agent.get_memory_snapshot()

        log_entry = {
            "session_id": sess.get("session_id", f"session_{i}"),
            "session_type": sess.get("type", "unknown"),
            "evidence_type": sess.get("evidence_type", ""),
            "ingest_result": {k: v for k, v in result.items() if k != "token_usage"},
            "token_usage": result.get("token_usage", {}),
            "ingest_time_sec": round(ingest_elapsed, 2),
            "memory_snapshot": snapshot["text"],
        }
        ingest_logs.append(log_entry)

        count = i - from_index + 1
        total = to_index - from_index
        sess_type = sess.get("type", "?")
        detail = ""
        if result:
            for k in ["memory_entries", "user_entries", "pages", "sessions_total"]:
                if k in result:
                    detail += f" {k}={result[k]}"
        print(f"    [{count}/{total}] {sess.get('session_id','?')[:30]:30s} ({sess_type:8s}){detail}")

    return ingest_logs


def ask_questions(agent: BaseMemorySystem, questions: List[Dict],
                  client=None, model=None) -> List[Dict]:
    """Ask questions and collect answers + retrieved context."""
    results = []
    for i, q in enumerate(questions):
        t0 = time.time()
        answer = agent.answer_question(q["question"], client=client, model=model)
        answer_elapsed = time.time() - t0
        retrieved = agent.get_retrieved_context()
        result = {**q}
        result["agent_answer"] = answer
        result["retrieved_context"] = retrieved
        result["answer_time_sec"] = round(answer_elapsed, 2)
        results.append(result)
        print(f"    Q{i+1}/{len(questions)}: {q['question'][:60]}  → {answer[:60]}... ({answer_elapsed:.1f}s)")
    return results


def run_episode(agent: BaseMemorySystem, episode: Dict, answer_client=None, answer_model=None) -> Dict:
    """Run one full episode: feed → before-questions → feed → after-questions.

    answer_client/answer_model: unified LLM for answer phase (same for all systems).
    """
    agent.reset()

    # Set episode-specific namespace for systems that need it (Mem0, etc.)
    # Set episode-specific namespace for isolation
    import uuid as _uuid
    ep_namespace = f"ep_{episode['episode_id']}_{episode.get('domain', 'unknown')}_{_uuid.uuid4().hex[:8]}"
    if hasattr(agent, '_set_user_id'):
        agent._set_user_id(ep_namespace)
    if hasattr(agent, '_set_group_id'):
        agent._set_group_id(ep_namespace)

    sessions = episode["sessions"]
    before_q = episode["before_questions"]
    after_q = episode["after_questions"]

    before_pos = before_q["position_after_session"] + 1
    after_pos = after_q["position_after_session"] + 1

    # Phase 1: Feed sessions up to before-questions
    print(f"  Phase 1: Feeding sessions 0-{before_pos-1} ({before_pos} sessions)...")
    ingest_logs_1 = feed_sessions(agent, sessions, 0, before_pos)
    agent.finalize_ingest()
    # Update last session's snapshot to include finalized state (e.g., Karpathy compile)
    if ingest_logs_1:
        ingest_logs_1[-1]["memory_snapshot"] = agent.get_memory_snapshot()["text"]
    snapshot_before = agent.get_memory_snapshot()

    # Phase 2: Before-questions
    print(f"  Phase 2: Before-questions ({len(before_q['questions'])} questions)...")
    before_answers = ask_questions(agent, before_q["questions"],
                                  client=answer_client, model=answer_model)

    # Phase 3: Feed change/delete sessions
    print(f"  Phase 3: Feeding sessions {before_pos}-{after_pos-1} ({after_pos - before_pos} sessions)...")
    ingest_logs_2 = feed_sessions(agent, sessions, before_pos, after_pos)
    agent.finalize_ingest()
    if ingest_logs_2:
        ingest_logs_2[-1]["memory_snapshot"] = agent.get_memory_snapshot()["text"]
    snapshot_after = agent.get_memory_snapshot()

    # Phase 4: After-questions
    print(f"  Phase 4: After-questions ({len(after_q['questions'])} questions)...")
    after_answers = ask_questions(agent, after_q["questions"],
                                  client=answer_client, model=answer_model)

    # Collect gold facts from evidence sessions
    all_gold_facts = []
    for sess in sessions:
        if sess.get("type") == "evidence":
            all_gold_facts.extend(sess.get("gold_facts", []))

    # Aggregate token usage (ingest + answer)
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    for log in ingest_logs_1 + ingest_logs_2:
        u = log.get("token_usage", {})
        total_usage["input_tokens"] += u.get("input_tokens", 0)
        total_usage["output_tokens"] += u.get("output_tokens", 0)
    # Add unified answer token usage
    answer_usage = getattr(agent, '_answer_token_usage', {})
    total_usage["answer_input_tokens"] = answer_usage.get("input_tokens", 0)
    total_usage["answer_output_tokens"] = answer_usage.get("output_tokens", 0)
    total_usage["input_tokens"] += answer_usage.get("input_tokens", 0)
    total_usage["output_tokens"] += answer_usage.get("output_tokens", 0)
    # Add retrieve token usage (MD-flat tool loop)
    retrieve_usage = getattr(agent, '_retrieve_token_usage', {})
    total_usage["retrieve_input_tokens"] = retrieve_usage.get("input_tokens", 0)
    total_usage["retrieve_output_tokens"] = retrieve_usage.get("output_tokens", 0)
    total_usage["input_tokens"] += retrieve_usage.get("input_tokens", 0)
    total_usage["output_tokens"] += retrieve_usage.get("output_tokens", 0)

    result = {
        "episode_id": episode["episode_id"],
        "domain": episode["domain"],
        "root": episode.get("root", ""),
        "total_sessions_fed": after_pos,
        "token_usage": total_usage,
        "memory_snapshots": {
            "before_questions": snapshot_before["text"],
            "after_questions": snapshot_after["text"]
        },
        "ingest_logs": ingest_logs_1 + ingest_logs_2,
        "before_answers": before_answers,
        "after_answers": after_answers,
        "gold_facts": all_gold_facts
    }

    # Cleanup after episode (delete group data from DB to prevent memory buildup)
    agent.reset()

    return result


# ============================================================
# Single Episode Processor (for parallel execution)
# ============================================================

def run_chunk(worker_id, ep_paths, agent_type, model, api_key, output_dir, internal_model, neo4j_base_port, top_k=None, seed=None):
    """Run a chunk of episodes sequentially on one Neo4j instance."""
    port = (neo4j_base_port + worker_id) if neo4j_base_port else None
    if seed is not None:
        from eval.seed_injector import install_seed_patches
        install_seed_patches(seed)
    results = []
    for ep_path in ep_paths:
        r = process_one_episode(ep_path, agent_type, model, api_key, output_dir, internal_model, port, top_k=top_k)
        results.append(r)
    return results


def process_one_episode(ep_path, agent_type, model, api_key, output_dir, internal_model=None, neo4j_port=None, top_k=None):
    """Process one episode. Creates its own agent instance (process-safe)."""
    with open(ep_path) as f:
        episode = json.load(f)

    ep_id = episode["episode_id"]
    print(f"  Ep{ep_id:3d} START (root={episode.get('root','?')}, "
          f"{episode['total_sessions']} sessions, "
          f"~{episode.get('total_tokens_approx','?')} tokens)")

    extra_kwargs = {}
    if neo4j_port and agent_type == "graphiti":
        extra_kwargs["neo4j_uri"] = f"bolt://localhost:{neo4j_port}"
    if top_k is not None and agent_type in ("bm25", "dense", "mem0"):
        extra_kwargs["top_k"] = top_k
    agent = create_agent(agent_type, model=model, api_key=api_key, internal_model=internal_model, **extra_kwargs)

    # Create unified answer client (same LLM for all systems' answer phase)
    from agents.base import BaseMemorySystem
    if model.startswith("claude"):
        from agents.anthropic_adapter import AnthropicAsOpenAI
        answer_client = AnthropicAsOpenAI(api_key=api_key)
    else:
        answer_client = OpenAI(api_key=api_key)

    # Reset budget tracker for this episode (process-global counter)
    tracker = get_tracker()
    tracker.reset()

    t0 = time.time()
    output = run_episode(agent, episode, answer_client=answer_client, answer_model=model)
    elapsed = time.time() - t0

    output["config"] = {
        "agent_type": agent_type,
        "agent_model": model,
        "internal_model": internal_model,
        "top_k": top_k,
    }
    output["budget"] = tracker.snapshot()

    domain = episode["domain"]
    domain_prefix = {"personal_life": "pl", "software_project": "sw"}[domain]
    os.makedirs(output_dir, exist_ok=True)
    model_tag = model.replace("/", "-")
    out_path = os.path.join(output_dir, f"agent_{domain_prefix}_{ep_id:03d}_{agent_type}_{model_tag}.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    n_before = len(output["before_answers"])
    n_after = len(output["after_answers"])
    print(f"  Ep{ep_id:3d} DONE  ({elapsed:.0f}s) → {out_path} | "
          f"before_q={n_before} after_q={n_after}")

    return {"ep_id": ep_id, "elapsed": elapsed, "out_path": out_path,
            "before_q": n_before, "after_q": n_after}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Run agent on assembled episodes")
    parser.add_argument("-e", "--episode", type=str, default=None,
                        help="Path to a single assembled episode JSON")
    parser.add_argument("-d", "--episode_dir", type=str, default="../data/assembled",
                        help="Directory of assembled episodes")
    parser.add_argument("-o", "--output_dir", type=str, default="agent_outputs",
                        help="Output directory (default: agent_outputs/)")
    parser.add_argument("--agent-type", type=str, default="md_file",
                        choices=["md_file", "karpathy", "mem0", "graphiti", "bm25", "dense"],
                        help="Agent type (default: md_file)")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="Model for agent LLM (default: gpt-4.1-mini, matches paper main run)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--internal-model", type=str, default=None,
                        help="Override all systems' internal LLM (e.g., gpt-4.1-mini). If not set, each system uses its native default.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override retrieval depth for raw-retrieval agents (bm25, dense). If not set, each agent uses its native default (5).")
    parser.add_argument("--seed", type=int, default=None,
                        help="LLM seed (passed through to OpenAI/Anthropic). Used for the multi-seed stability ablation.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip episodes whose agent output JSON already exists in --output_dir.")
    args = parser.parse_args()

    if args.seed is not None:
        from eval.seed_injector import install_seed_patches
        install_seed_patches(args.seed)

    # Answer LLM always needs Anthropic key if model is claude
    if args.model.startswith("claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: Set ANTHROPIC_API_KEY environment variable")
            sys.exit(1)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: Set OPENAI_API_KEY environment variable")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.episode:
        episode_files = [args.episode]
    else:
        episode_files = sorted([
            os.path.join(args.episode_dir, f)
            for f in os.listdir(args.episode_dir)
            if f.startswith("episode_") and f.endswith(".json")
        ])

    if not episode_files:
        print("No episode files found.")
        sys.exit(1)

    # Skip episodes whose output already exists (resumable runs)
    if args.skip_existing:
        model_tag = args.model.replace("/", "-")
        kept = []
        skipped = 0
        for ep_path in episode_files:
            with open(ep_path) as _f:
                _ep = json.load(_f)
            _dom = _ep["domain"]
            _prefix = {"personal_life": "pl", "software_project": "sw"}[_dom]
            _out = os.path.join(args.output_dir,
                f"agent_{_prefix}_{_ep['episode_id']:03d}_{args.agent_type}_{model_tag}.json")
            if os.path.exists(_out):
                skipped += 1
            else:
                kept.append(ep_path)
        print(f"--skip-existing: {skipped} already done, {len(kept)} remaining")
        episode_files = kept
        if not episode_files:
            print("All episodes already processed. Nothing to do.")
            sys.exit(0)

    print(f"Running {args.agent_type} agent on {len(episode_files)} episodes "
          f"(model={args.model}, workers={args.workers})\n")

    t_start = time.time()
    summaries = []

    internal_model = args.internal_model

    # For Graphiti parallel: each worker gets a dedicated Neo4j port
    neo4j_base_port = int(os.environ.get("NEO4J_BASE_PORT", "0"))  # 0 = use default 7687

    if args.workers == 1:
        for ep_path in episode_files:
            port = neo4j_base_port if neo4j_base_port else None
            summary = process_one_episode(
                ep_path, args.agent_type, args.model, api_key, args.output_dir,
                internal_model=internal_model, neo4j_port=port, top_k=args.top_k)
            summaries.append(summary)
    else:
        # Chunk episodes by worker so each worker processes its chunk sequentially
        # This ensures one Neo4j instance is never shared simultaneously
        chunks = [[] for _ in range(args.workers)]
        for i, ep_path in enumerate(episode_files):
            chunks[i % args.workers].append(ep_path)

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_chunk, wid, chunk, args.agent_type,
                               args.model, api_key, args.output_dir,
                               internal_model, neo4j_base_port,
                               args.top_k, args.seed): wid
                for wid, chunk in enumerate(chunks) if chunk
            }
            for future in as_completed(futures):
                wid = futures[future]
                try:
                    chunk_results = future.result()
                    summaries.extend(chunk_results)
                except Exception as e:
                    print(f"  ERROR: worker {wid} — {e}")

    total_elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"DONE: {len(summaries)}/{len(episode_files)} episodes in {total_elapsed:.0f}s")
    if summaries:
        avg = sum(s["elapsed"] for s in summaries) / len(summaries)
        print(f"  Avg per episode: {avg:.0f}s (wall clock: {total_elapsed:.0f}s)")
    print(f"  Outputs in {args.output_dir}/")
    print(f"  Next: python3 judge.py -d {args.output_dir}")
