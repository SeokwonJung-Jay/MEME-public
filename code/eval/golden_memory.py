"""
Golden Memory Oracle — Task solvability upper bound.

Bypasses memory system W/R entirely. Injects gold facts directly as context,
then uses the SAME unified answer prompt + LLM to answer.

Measures: "If the correct facts are in context, can the LLM answer correctly?"

Usage:
  python3 golden_memory.py -e ../data/v6/filler_pl/episode_001.json -g ../data/v6/gold_facts_pl/gold_facts_001.json
  python3 golden_memory.py --episode-dir ../data/v6/filler_pl/ --gold-dir ../data/v6/gold_facts_pl/ -o golden_outputs/
"""

import json
import os
import sys
import argparse

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai required")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agents.base import UNIFIED_ANSWER_PROMPT


# ============================================================
# Gold context builders — one per task type
# ============================================================

def build_exact_context(task, gold_facts):
    """Exact recall: inject the gold verbatim text."""
    gold_value = task["gold_answer"]
    entity = (task.get("target_entities") or task.get("entity", [None]))[0]
    return f"User asked you to remember this exactly:\n\"{gold_value}\""


def build_multiple_update_context(task, gold_facts, episode):
    """Tr: inject full history in order."""
    entity = (task.get("target_entities") or task.get("entity", [None]))[0]
    history = task["entity_values"].get(entity, [])
    if isinstance(history, str):
        history = [h.strip() for h in history.split(",")]

    lines = []
    for i, val in enumerate(history, 1):
        lines.append(f"{i}. {val}")

    entity_name = entity.replace("_", " ")
    return f"User's {entity_name} history (in chronological order):\n" + "\n".join(lines)


def build_multihop_context(task, gold_facts):
    """Agg: inject all entity values."""
    entity_values = task["entity_values"]
    lines = []
    for entity, value in entity_values.items():
        entity_name = entity.replace("_", " ")
        lines.append(f"- User's {entity_name}: {value}")
    return "Related information:\n" + "\n".join(lines)


def build_cascade_d_context(task, gold_facts, episode):
    """Cas: root change + dependent before value + if-then rule + new dependent value."""
    entity = (task.get("target_entities") or task.get("entity", [None]))[0]
    gold_value = task["gold_answer"]
    cascade_source = task.get("cascade_source", "")
    root = episode.get("root", "") or gold_facts.get("root", "")

    # Find root change info
    root_info = gold_facts.get("root_change", {}) or episode.get("root_change", {})
    root_before = root_info.get("before", "")
    root_after = root_info.get("after", "")

    # Find dependent entity's before value from phase3 (before questions)
    dep_before = ""
    for q in gold_facts.get("phase2_before_questions", []):
        q_entity = q.get("entity", [])
        if isinstance(q_entity, list) and entity in q_entity:
            dep_before = q.get("expected_answer", "")
            break
        elif q_entity == entity:
            dep_before = q.get("expected_answer", "")
            break

    # Find if-then rule from gold_facts phase1
    if_then_text = ""
    for fact in gold_facts.get("phase1_fact_introduction", []):
        if fact.get("is_if_then") and fact.get("entity") == entity:
            if_then_text = fact["gold_fact"]
            break

    # Find dependency declaration from gold_facts phase1
    dep_declaration = ""
    for fact in gold_facts.get("phase1_fact_introduction", []):
        if fact.get("has_dependency") and fact.get("entity") == entity and not fact.get("is_if_then"):
            dep_declaration = fact["gold_fact"]
            break

    lines = []
    root_name = root.replace("_", " ")
    entity_name = entity.replace("_", " ")
    lines.append(f"- User's {root_name} changed from '{root_before}' to '{root_after}'.")
    if dep_before:
        lines.append(f"- User's {entity_name} was previously '{dep_before}'.")
    if dep_declaration:
        lines.append(f"- Dependency: \"{dep_declaration}\"")
    if if_then_text:
        lines.append(f"- Rule: \"{if_then_text}\"")
    lines.append(f"- User's current {entity_name}: {gold_value}")

    return "Related information:\n" + "\n".join(lines)


def build_cascade_u_context(task, gold_facts, episode):
    """Abs: root change + dependency + old value. NO new value (LLM must express uncertainty)."""
    entity = (task.get("target_entities") or task.get("entity", [None]))[0]
    cascade_source = task.get("cascade_source", "")
    root = episode.get("root", "") or gold_facts.get("root", "")

    root_info = gold_facts.get("root_change", {}) or episode.get("root_change", {})
    root_before = root_info.get("before", "")
    root_after = root_info.get("after", "")

    # Find dependency declaration from gold_facts
    dep_text = ""
    for fact in gold_facts.get("phase1_fact_introduction", []):
        if fact.get("has_dependency") and fact.get("entity") == entity and not fact.get("is_if_then"):
            dep_text = fact["gold_fact"]
            break

    # Get the old value
    old_value = task["entity_values"].get(entity, "")

    lines = []
    root_name = root.replace("_", " ")
    entity_name = entity.replace("_", " ")
    lines.append(f"- User's {root_name} changed from '{root_before}' to '{root_after}'.")
    if dep_text:
        lines.append(f"- Previously stated: \"{dep_text}\"")
    lines.append(f"- User's {entity_name} was previously '{old_value}', but no new value was declared after the change.")
    lines.append(f"- Since {root_name} changed and {entity_name} depended on it, the previous value may no longer be valid.")

    return "Related information:\n" + "\n".join(lines)


def build_deletion_context(task, gold_facts):
    """Deletion: deleted value + delete instruction."""
    entity = (task.get("target_entities") or task.get("entity", [None]))[0]
    deleted_value = task["entity_values"].get(entity, "")

    # Find delete instruction from gold_facts
    delete_text = ""
    for fact in gold_facts.get("phase3_change_and_deletion", []):
        if fact.get("entity") == entity and "delete" in fact.get("type", "").lower():
            delete_text = fact["gold_fact"]
            break

    entity_name = entity.replace("_", " ")
    lines = []
    lines.append(f"- User previously mentioned their {entity_name}: '{deleted_value}'.")
    if delete_text:
        lines.append(f"- User later requested: \"{delete_text}\"")
    else:
        lines.append(f"- User later requested to delete/forget this information.")

    return "Related information:\n" + "\n".join(lines)


# ============================================================
# Main: run golden memory for one episode
# ============================================================

CONTEXT_BUILDERS = {
    "ER": build_exact_context,
    "Tr": build_multiple_update_context,
    "Agg": build_multihop_context,
    "Cas": build_cascade_d_context,
    "Abs": build_cascade_u_context,
    "Del": build_deletion_context,
}


def _build_filler_text(episode):
    """Extract all filler session texts from assembled episode."""
    parts = []
    for sess in episode.get("sessions", []):
        if sess.get("type") == "filler":
            for turn in sess.get("conversation", []):
                role = "User" if turn["role"] == "user" else "Assistant"
                parts.append(f"[{role}]: {turn['content']}")
            parts.append("")  # separator between sessions
    return "\n".join(parts)


def run_golden_memory(episode, gold_facts, client, model, with_filler=False):
    """Run golden memory oracle for one episode."""
    after_questions = episode.get("after_questions", {}).get("questions", [])
    before_questions = episode.get("before_questions", {}).get("questions", [])

    filler_text = _build_filler_text(episode) if with_filler else ""

    results = {
        "episode_id": episode["episode_id"],
        "domain": episode.get("domain", ""),
        "root": episode.get("root", ""),
        "config": {"agent_type": "golden_memory", "agent_model": model},
        "memory_snapshots": {"before_questions": "(golden)", "after_questions": "(golden)"},
        "ingest_logs": [],
        "before_answers": [],
        "after_answers": [],
        "gold_facts": [],
    }

    # Before questions: skip in oracle mode (solvability = after questions only)

    # After questions: task-specific gold context
    for q in after_questions:
        task_base = q["task_type"].split(" (")[0]
        builder = CONTEXT_BUILDERS.get(task_base)

        if builder is None:
            print(f"  WARNING: No context builder for {task_base}, skipping")
            continue

        # Build gold context
        if task_base in ("Cas", "Abs", "Tr"):
            context = builder(q, gold_facts, episode)
        else:
            context = builder(q, gold_facts)

        # Append filler noise if requested
        if filler_text:
            context = context + "\n\nOther conversation history:\n" + filler_text

        prompt = UNIFIED_ANSWER_PROMPT.format(context=context, question=q["question"])
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()

        results["after_answers"].append({
            **q,
            "agent_answer": answer,
            "retrieved_context": context,
        })
        print(f"  After [{q['task_type']}] → {answer[:60]}...")

    return results


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golden Memory Oracle")
    parser.add_argument("-e", "--episode", type=str, help="Single assembled episode JSON")
    parser.add_argument("-g", "--gold-facts", type=str, help="Gold facts JSON for the episode")
    parser.add_argument("--episode-dir", type=str, help="Directory of assembled episodes")
    parser.add_argument("--gold-dir", type=str, help="Directory of gold facts")
    parser.add_argument("-o", "--output-dir", type=str, default="golden_outputs")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--with-filler", action="store_true",
                        help="Append filler session texts after gold context (noise test)")
    args = parser.parse_args()

    # Route by model: Anthropic models → anthropic_adapter; OpenAI models → OpenAI client.
    if args.model.startswith("claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: Set ANTHROPIC_API_KEY")
            sys.exit(1)
        from agents.anthropic_adapter import AnthropicAsOpenAI
        client = AnthropicAsOpenAI(api_key=api_key)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: Set OPENAI_API_KEY")
            sys.exit(1)
        client = OpenAI()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect episode + gold_facts pairs
    pairs = []
    if args.episode and args.gold_facts:
        pairs.append((args.episode, args.gold_facts))
    elif args.episode_dir and args.gold_dir:
        ep_files = sorted([f for f in os.listdir(args.episode_dir) if f.startswith("episode_") and f.endswith(".json")])
        for ef in ep_files:
            ep_id = ef.replace("episode_", "").replace(".json", "")
            gf = f"gold_facts_{ep_id}.json"
            gf_path = os.path.join(args.gold_dir, gf)
            if os.path.exists(gf_path):
                pairs.append((os.path.join(args.episode_dir, ef), gf_path))
    else:
        print("Provide -e/-g or --episode-dir/--gold-dir")
        sys.exit(1)

    print(f"Running Golden Memory on {len(pairs)} episodes (model={args.model})\n")

    for ep_path, gf_path in pairs:
        with open(ep_path) as f:
            episode = json.load(f)
        with open(gf_path) as f:
            gold_facts = json.load(f)

        ep_id = episode["episode_id"]
        domain = episode.get("domain", "unknown")
        domain_prefix = {"personal_life": "pl", "software_project": "sw"}.get(domain, domain)

        print(f"Ep{ep_id} ({domain_prefix}, root={episode.get('root','')})")
        result = run_golden_memory(episode, gold_facts, client, args.model,
                                   with_filler=args.with_filler)

        out_file = f"golden_{domain_prefix}_{ep_id:03d}_{args.model.replace('/', '-')}.json"
        with open(os.path.join(args.output_dir, out_file), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        n_before = len(result["before_answers"])
        n_after = len(result["after_answers"])
        print(f"  Done: {n_before} before + {n_after} after → {out_file}\n")

    print("Golden Memory complete.")
