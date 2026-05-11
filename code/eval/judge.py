"""
Judge — task-specific LLM correctness check on agent outputs.

Reads agent_output JSONs and judges each (question, agent_answer, gold_value) using a
per-task prompt (ER, Agg, Tr, Del, Cas, Abs). For Cas/Abs/Del, applies the trivial-pass
filter that distinguishes real passes from "I don't know" lucky matches.

Separate from run_agent.py so that:
  - Agent run (expensive) only happens once
  - Judge can be re-run with different judge models
  - Agent errors do not lose already-collected answers

Usage:
  python3 judge.py -d agent_outputs/
  python3 judge.py -e agent_outputs/agent_005.json
  python3 judge.py -d agent_outputs/ --judge-model gpt-4o-mini
"""

import json
import os
import sys
import argparse
from typing import List, Dict

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai required")
    sys.exit(1)


# ============================================================
# LLM Judge
# ============================================================

class LLMJudge:
    """LLM-based judge for per-task answer correctness.

    Each (entity, value) pair is evaluated against the agent's answer using a
    task-specific prompt. For Tr the gold is a list of ordered values. For Agg
    a single LLM call checks all entity values at once (u_check_multi).
    """

    def __init__(self, client, model="gpt-4o", max_retries=3):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def _call(self, prompt: str) -> Dict:
        import time
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                self.total_calls += 1
                if hasattr(response, 'usage') and response.usage:
                    self.total_input_tokens += getattr(response.usage, 'prompt_tokens', 0) or 0
                    self.total_output_tokens += getattr(response.usage, 'completion_tokens', 0) or 0
                content = response.choices[0].message.content
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    content = content.strip()
                    if content.startswith("```"):
                        content = content.split("\n", 1)[1]
                        if content.endswith("```"):
                            content = content[:-3]
                    return json.loads(content)
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    # ----------------------------------------------------------
    # Task-specific prompts
    # ----------------------------------------------------------

    BEFORE_COMMON_PROMPT = """You are a judge evaluating whether an AI agent's answer is semantically correct.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

Does the agent's answer contain the correct information matching the gold answer?

Rules:
- Focus on semantic equivalence, not exact wording
- If the gold value is present in the agent's answer, it is correct — regardless of any additional information, future possibilities, or extra details the agent mentions
- "Dentist every 6 months; dermatologist monthly (if you change residence)" → gold is "dentist (every 6 months)" → YES (core answer correct, extra info is irrelevant)
- "40 minutes" = "40 min" → YES
- "dentist appointment every 6 months" = "dentist (every 6 months)" → YES
- If agent says "I don't know" and gold is a specific value → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    CASCADE_D_PROMPT = """You are a judge evaluating whether an AI agent correctly reports a CASCADE-DEPENDENT value after an upstream change.

QUESTION: {question}
GOLD (new cascaded value): {gold}
AGENT: {agent_answer}

The upstream entity changed, triggering a cascade. The agent should now report ONLY the new value.

Rules:
- Agent states the new value as the sole, definitive answer → YES
- Agent lists BOTH old and new values together → NO
- Agent states only the old value → NO
- Agent lists multiple options or hedges ("might", "considering") → NO
- Agent uses future tense ("will change to X", "will be X") instead of confirming as current → NO
- "I don't know" → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    CASCADE_U_PROMPT = """You are a judge evaluating whether an AI agent appropriately handles UNCERTAINTY after an upstream entity changed.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

The upstream entity changed, making the previous answer potentially invalid. The agent should express uncertainty.

Rules:
- Agent says "I don't know" / "not sure" / "Unknown" / "None" / expresses uncertainty → YES
- Agent acknowledges the upstream change and questions validity → YES
- Agent confidently states the OLD value as definitive current fact → NO
- Agent provides the old value with no hedging or uncertainty → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    DELETION_PROMPT = """You are a judge evaluating whether an AI agent correctly handles a DELETED piece of information.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

The information was explicitly deleted/removed. The agent should not return it.

Rules:
- Agent says "I don't have that information" / "no longer available" / "Unknown" / "None" / similar WITHOUT revealing the deleted value → YES
- Agent indicates the item no longer exists or applies WITHOUT revealing the deleted value (e.g., "No food allergies" when the deleted value was a specific allergy) → YES
- Agent reveals the deleted value in any way (even while saying it was deleted/removed) → NO
- Agent returns the deleted value as if it still exists → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    MULTIHOP_PROMPT = """You are a judge evaluating whether an AI agent's answer contains specific values.

QUESTION: {question}
TARGET VALUES: {gold_values}
AGENT: {agent_answer}

For EACH target value, check whether it appears (exact or paraphrased) in the agent's answer.

Rules:
- The value appears in the answer (exact or paraphrased) → true
- The value is absent → false
- "I don't know" or "I don't have that information" → all false

Output ONLY JSON: {{"results": [{{"value": "...", "present": true/false}}], "reason": "one sentence"}}"""

    MULTIPLE_UPDATE_PROMPT = """You are a judge evaluating whether an AI agent correctly recalls a sequence of historical values for an entity that changed multiple times.

QUESTION: {question}
GOLD (ordered history values, comma-separated): {gold}
AGENT: {agent_answer}

The user's information changed multiple times. The question asks for the FULL HISTORY in chronological order (earliest to latest).

Rules:
- Agent's answer must contain ALL gold values in the EXACT order specified
- Order matters: the values must appear earliest-to-latest as in the gold
- Extra surrounding text is fine, but the gold sequence must be preserved
- Missing any value → NO
- Wrong order → NO
- Only some values → NO
- "I don't know" → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    EXACT_PROMPT = """You are a judge evaluating whether an AI agent correctly recalls an EXACT value verbatim.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

Does the agent's answer contain the exact value?

Rules:
- Check if the gold value appears VERBATIM as a substring in the agent's answer
- If the gold value is fully contained in the answer, even with extra words before/after it → YES
- Example: gold="OOM killed by container runtime", agent="OOM killed by container runtime errors" → YES (gold is fully preserved)
- Minor formatting differences are acceptable (e.g., extra spaces, capitalization)
- Missing or substituted words WITHIN the gold value → NO
- "I don't know" or "I don't have that information" → NO
- A completely different value → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    JUDGE_PROMPTS = {
        "Tr": MULTIPLE_UPDATE_PROMPT,
        "Cas": CASCADE_D_PROMPT,
        "Abs": CASCADE_U_PROMPT,
        "Del": DELETION_PROMPT,
        "Agg": MULTIHOP_PROMPT,
        "ER": EXACT_PROMPT,
    }

    # ----------------------------------------------------------
    # Per-question correctness check
    # ----------------------------------------------------------

    def u_check(self, question: str, entity: str, gold_value, agent_answer: str,
                task_type: str, phase: str = "after") -> Dict:
        """Task-specific correctness judge. phase=before uses BEFORE_COMMON_PROMPT
        for all non-Agg tasks; phase=after uses task-specific prompts."""
        task_base = task_type.split(" (")[0]

        # ER: deterministic substring match (no LLM call)
        if task_base == "ER":
            gold_norm = ' '.join(gold_value.lower().split())
            agent_norm = ' '.join(agent_answer.lower().split())
            matched = gold_norm in agent_norm
            return {
                "u_pass": matched,
                "u_reason": "Gold value found in answer (substring match)" if matched
                            else "Gold value not found in answer (substring match)",
            }

        # Tr: gold_value is an ordered list of historical values
        if task_base == "Tr":
            history = gold_value if isinstance(gold_value, list) else [gold_value]
            gold_str = ", ".join(history)
            prompt = self.JUDGE_PROMPTS["Tr"].format(
                question=question, gold=gold_str, agent_answer=agent_answer
            )
            result = self._call(prompt)
            agent_lower = agent_answer.lower()
            positions = [agent_lower.find(v.lower()) for v in history]
            partial_pass = 0
            last_pos = -1
            for pos in positions:
                if pos > last_pos:
                    partial_pass += 1
                    last_pos = pos
            return {
                "u_pass": result.get("correct", False),
                "u_reason": result.get("reason", ""),
                "u_pass_count": partial_pass,
                "u_pass_total": len(history),
            }

        if phase == "before" and task_base != "Agg":
            template = self.BEFORE_COMMON_PROMPT
        else:
            template = self.JUDGE_PROMPTS[task_base]

        prompt = template.format(question=question, gold=gold_value, agent_answer=agent_answer)
        result = self._call(prompt)
        return {"u_pass": result.get("correct", False), "u_reason": result.get("reason", "")}

    def u_check_multi(self, question: str, entity_values: Dict[str, str],
                      agent_answer: str, phase: str = "after") -> Dict:
        """Agg correctness: single LLM call checks all entity values; pass requires all."""
        entity_order = list(entity_values.keys())
        values_list = [entity_values[e] for e in entity_order]
        gold_values_str = ", ".join(f'"{v}"' for v in values_list)

        prompt = self.MULTIHOP_PROMPT.format(
            question=question, gold_values=gold_values_str, agent_answer=agent_answer
        )
        result = self._call(prompt)

        per_entity = {}
        results_list = result.get("results", [])
        for i, entity in enumerate(entity_order):
            per_entity[entity] = results_list[i].get("present", False) if i < len(results_list) else False

        pass_count = sum(1 for v in per_entity.values() if v)
        total = len(per_entity)
        missing = [e for e, v in per_entity.items() if not v]
        return {
            "u_pass": pass_count == total,
            "u_pass_per_entity": per_entity,
            "u_pass_count": pass_count,
            "u_pass_total": total,
            "u_reason": result.get("reason",
                                   f"Missing: {', '.join(missing)}" if missing else "All entities correct"),
        }


# ============================================================
# Judge one agent output
# ============================================================

def judge_episode(agent_output: Dict, judge: LLMJudge, max_workers: int = 8) -> Dict:
    """Run U-check on one episode and apply trivial-pass classification.

    Per-question pass/fail is from u_check (non-Agg) or u_check_multi (Agg).
    Cas/Abs/Del get a `pass_type` field in {real, trivial, knew_but_failed, never_knew}
    derived from the same entity's before-question answer, per the trivial-pass rule.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    before_answers = agent_output.get("before_answers", [])
    after_answers = agent_output.get("after_answers", [])

    tasks = {}

    # before-questions: skip ER (already trivially correct) — others run for trivial-pass cross-check
    for i, ba in enumerate(before_answers):
        tt = ba.get("task_type", "")
        task_base = tt.split(" (")[0]
        if task_base == "ER":
            continue
        ev = ba["entity_values"]
        entity = list(ev.keys())[0]
        value = ev[entity]
        tasks[("before", i)] = lambda q=ba["question"], e=entity, v=value, a=ba.get("agent_answer", ""), t=tt: judge.u_check(
            question=q, entity=e, gold_value=v, agent_answer=a, task_type=t, phase="before"
        )

    # after-questions: task-specific
    for i, aa in enumerate(after_answers):
        ev = aa["entity_values"]
        tt = aa.get("task_type", "")
        task_base = tt.split(" (")[0]
        if task_base == "Agg":
            tasks[("after", i)] = lambda q=aa["question"], evs=ev, a=aa.get("agent_answer", ""): judge.u_check_multi(
                question=q, entity_values=evs, agent_answer=a, phase="after"
            )
        else:
            entity = list(ev.keys())[0]
            value = ev[entity]
            tasks[("after", i)] = lambda q=aa["question"], e=entity, v=value, a=aa.get("agent_answer", ""), t=tt: judge.u_check(
                question=q, entity=e, gold_value=v, agent_answer=a, task_type=t, phase="after"
            )

    print(f"  Running {len(tasks)} judge checks (workers={max_workers})...")
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            results[key] = future.result()

    # Reassemble per-answer records
    before = []
    for i, ba in enumerate(before_answers):
        r = results.get(("before", i), {"u_pass": False, "u_reason": "skipped (ER)"})
        before.append({**ba, **r})

    after = []
    for i, aa in enumerate(after_answers):
        r = results.get(("after", i), {"u_pass": False, "u_reason": "missing"})
        after.append({**aa, **r})

    # Trivial-pass classification for Cas/Abs/Del
    before_map = {}
    for ba in before:
        ent = list(ba["entity_values"].keys())[0]
        before_map[ent] = ba.get("u_pass", False)

    trivial_analysis = {}
    for task_key in ("Cas", "Abs", "Del"):
        counts = {"total": 0, "real_pass": 0, "trivial_pass": 0,
                  "knew_but_failed": 0, "never_knew": 0}
        for au in after:
            if au["task_type"].split(" (")[0] != task_key:
                continue
            counts["total"] += 1
            ent = list(au["entity_values"].keys())[0]
            b_pass = before_map.get(ent, False)
            a_pass = au.get("u_pass", False)
            if a_pass and b_pass:
                counts["real_pass"] += 1
                au["pass_type"] = "real"
            elif a_pass and not b_pass:
                counts["trivial_pass"] += 1
                au["pass_type"] = "trivial"
            elif not a_pass and b_pass:
                counts["knew_but_failed"] += 1
                au["pass_type"] = "knew_but_failed"
            else:
                counts["never_knew"] += 1
                au["pass_type"] = "never_knew"
        trivial_analysis[task_key] = counts

    return {
        "episode_id": agent_output["episode_id"],
        "root": agent_output.get("root", ""),
        "agent_config": agent_output.get("config", {}),
        "before_answers": before,
        "after_answers": after,
        "trivial_analysis": trivial_analysis,
        "totals": {
            "before_pass": sum(1 for r in before if r.get("u_pass")),
            "before_total": len(before),
            "after_pass": sum(1 for r in after if r.get("u_pass")),
            "after_total": len(after),
        },
    }


# ============================================================
# Main
# ============================================================

def process_one_judge(fp, api_key, judge_model, output_dir, check_workers):
    import time as _time
    client = OpenAI(api_key=api_key)
    judge = LLMJudge(client=client, model=judge_model)

    with open(fp) as f:
        agent_output = json.load(f)

    ep_id = agent_output["episode_id"]
    domain = agent_output.get("domain", "unknown")
    domain_prefix = {"personal_life": "pl", "software_project": "sw"}.get(domain, domain)
    agent_config = agent_output.get("config", {})
    agent_type = agent_config.get("agent_type", "unknown")
    agent_model_tag = agent_config.get("agent_model", "unknown").replace("/", "-")

    t0 = _time.time()
    result = judge_episode(agent_output, judge, max_workers=check_workers)
    result["judge_config"] = {"judge_model": judge_model}
    result["judge_usage"] = {
        "calls": judge.total_calls,
        "input_tokens": judge.total_input_tokens,
        "output_tokens": judge.total_output_tokens,
    }

    judge_model_tag = judge_model.replace("/", "-")
    out_path = os.path.join(output_dir,
        f"eval_{domain_prefix}_{ep_id:03d}_{agent_type}_{agent_model_tag}_{judge_model_tag}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    elapsed = _time.time() - t0
    t = result["totals"]
    print(f"  Ep{ep_id:3d} ({elapsed:.0f}s) | "
          f"before {t['before_pass']}/{t['before_total']}  after {t['after_pass']}/{t['after_total']}")
    return result


if __name__ == "__main__":
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Judge agent outputs (per-task LLM correctness)")
    parser.add_argument("-e", "--agent_output", type=str, default=None,
                        help="Path to a single agent output JSON")
    parser.add_argument("-d", "--agent_output_dir", type=str, default="agent_outputs",
                        help="Directory of agent outputs (default: agent_outputs/)")
    parser.add_argument("-o", "--output_dir", type=str, default="eval_results",
                        help="Output directory (default: eval_results/)")
    parser.add_argument("--judge-model", type=str, default="gpt-4o",
                        help="Judge LLM (default: gpt-4o)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Parallel episode workers (default: 4)")
    parser.add_argument("--check-workers", type=int, default=8,
                        help="Parallel check workers per episode (default: 8)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: Set OPENAI_API_KEY environment variable")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.agent_output:
        files = [args.agent_output]
    else:
        prefixes = ("agent_", "golden_")
        files = sorted([
            os.path.join(args.agent_output_dir, f)
            for f in os.listdir(args.agent_output_dir)
            if f.startswith(prefixes) and f.endswith(".json")
        ])

    if not files:
        sys.exit("No agent output files found.")

    print(f"Judging {len(files)} agent outputs (judge: {args.judge_model}, "
          f"workers={args.workers}, check_workers={args.check_workers})\n")

    t_start = time.time()
    all_results = []

    if args.workers == 1:
        for fp in files:
            all_results.append(process_one_judge(
                fp, api_key, args.judge_model, args.output_dir, args.check_workers))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one_judge, fp, api_key,
                                args.judge_model, args.output_dir, args.check_workers): fp
                for fp in files
            }
            for future in as_completed(futures):
                all_results.append(future.result())

    total_elapsed = time.time() - t_start

    if all_results:
        bp = sum(r["totals"]["before_pass"] for r in all_results)
        bt = sum(r["totals"]["before_total"] for r in all_results)
        ap = sum(r["totals"]["after_pass"] for r in all_results)
        at = sum(r["totals"]["after_total"] for r in all_results)
        print(f"\n{'='*60}")
        print(f"GRAND TOTAL ({len(all_results)} episodes, {total_elapsed:.0f}s)")
        print(f"  before: {bp}/{bt}  ({bp/bt*100:.1f}%)" if bt else "  before: 0/0")
        print(f"  after:  {ap}/{at}  ({ap/at*100:.1f}%)" if at else "  after:  0/0")

        for task_key in ("Cas", "Abs", "Del"):
            agg = {"total": 0, "real_pass": 0, "trivial_pass": 0,
                   "knew_but_failed": 0, "never_knew": 0}
            for r in all_results:
                ta = r.get("trivial_analysis", {}).get(task_key, {})
                for k in agg:
                    agg[k] += ta.get(k, 0)
            if agg["total"] > 0:
                tp = agg["real_pass"] + agg["trivial_pass"]
                print(f"  {task_key}: {tp}/{agg['total']} pass "
                      f"(real={agg['real_pass']} trivial={agg['trivial_pass']} "
                      f"knew_but_failed={agg['knew_but_failed']} never_knew={agg['never_knew']})")
