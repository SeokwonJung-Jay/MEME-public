"""Metric for SIMBA: U-check score with trivial-pass aware breakdown.

Design:
- score (optimization signal): raw u-pass average. Matches paper Sec 4.3 SIMBA.
- info: human-readable per-task breakdown, including trivial-pass classification
  for Cas/Abs/Del. Read by prompt_model (OfferFeedback) for richer advice.
- per_task / trivial_analysis: structured dicts logged by simba_patches for
  post-hoc analysis (also enable building paper Table 13-style trivial reports).

Trivial-pass classification (Cas/Abs/Del only):
  before=PASS, after=PASS → real            (system knew AND handled change)
  before=FAIL, after=PASS → trivial         ('I don't know' = lucky pass)
  before=PASS, after=FAIL → knew_but_failed (stored but didn't handle change)
  before=FAIL, after=FAIL → never_knew

Scope:
- After: all tasks (ER, Agg, Tr, Cas, Abs, Del)
- Before: all tasks except ER (ER-before redundant with ER-after)

Uses release/code/eval/judge.py (paper-tag labels: ER/Agg/Tr/Del/Cas/Abs).
Data fed to this metric must already be normalized to those tags
(use experiments/normalize_task_labels.py if not).
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

from openai import OpenAI

# Find judge.py — prefer release version, fall back to working copy.
_here = Path(__file__).resolve().parent
_candidates = [_here.parent.parent / "eval"]
for _p in _candidates:
    if (_p / "judge.py").exists():
        sys.path.insert(0, str(_p))
        _JUDGE_PATH = _p / "judge.py"
        break
else:
    raise RuntimeError(f"judge.py not found in any of: {_candidates}")

from judge import LLMJudge


_JUDGE_CLIENT = None
_JUDGE = None


def _get_judge():
    global _JUDGE_CLIENT, _JUDGE
    if _JUDGE is None:
        _JUDGE_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        _JUDGE = LLMJudge(client=_JUDGE_CLIENT, model="gpt-4o")
    return _JUDGE


def _entity_key(ans):
    """Stable key for cross-phase entity matching.
    Cas/Abs/Del are single-entity; entity_values dict has exactly 1 key.
    Agg is multi-entity but isn't in trivial-pass scope (skipped there).
    """
    ev = ans.get("entity_values", {}) or {}
    if ev:
        return list(ev.keys())[0]
    ent = ans.get("entity")
    if isinstance(ent, list):
        return ent[0] if ent else ""
    return ent or ""


def u_check_score(example, prediction, trace=None):
    import dspy

    judge = _get_judge()

    # =========================================================
    # After phase — all tasks
    # =========================================================
    after_results = []                          # (task_base, entity_key, passed)
    after_per_task = defaultdict(lambda: [0, 0])

    for ans in prediction.after_answers:
        task_type = str(ans["task_type"])
        task_base = task_type.split(" (")[0]

        if task_base == "Agg":
            result = judge.u_check_multi(
                question=ans["question"],
                entity_values=ans.get("entity_values", {}) or {},
                agent_answer=ans["agent_answer"],
            )
        else:
            ent = ans.get("entity")
            if isinstance(ent, list):
                ent = ent[0] if ent else ""
            result = judge.u_check(
                question=ans["question"],
                entity=ent,
                gold_value=ans["expected_answer"],
                agent_answer=ans["agent_answer"],
                task_type=task_type,
                phase="after",
            )

        passed = bool(result.get("u_pass", False))
        after_per_task[task_base][1] += 1
        if passed:
            after_per_task[task_base][0] += 1
        after_results.append((task_base, _entity_key(ans), passed))

    # =========================================================
    # Before phase — skip ER (redundant with ER-after)
    # =========================================================
    before_per_task = defaultdict(lambda: [0, 0])
    before_pass_by_entity = {}                  # entity_key → bool, for trivial-pass

    for ans in prediction.before_answers:
        task_type = str(ans["task_type"])
        task_base = task_type.split(" (")[0]
        if task_base == "ER":
            continue

        ent = ans.get("entity")
        if isinstance(ent, list):
            ent = ent[0] if ent else ""
        result = judge.u_check(
            question=ans["question"],
            entity=ent,
            gold_value=ans["expected_answer"],
            agent_answer=ans["agent_answer"],
            task_type=task_type,
            phase="before",
        )
        passed = bool(result.get("u_pass", False))
        before_per_task[task_base][1] += 1
        if passed:
            before_per_task[task_base][0] += 1
        before_pass_by_entity[_entity_key(ans)] = passed

    # =========================================================
    # Trivial-pass classification (Cas/Abs/Del × before/after)
    # =========================================================
    trivial_analysis = {}
    for task_key in ("Cas", "Abs", "Del"):
        counts = {"total": 0, "real": 0, "trivial": 0,
                  "knew_failed": 0, "never_knew": 0}
        for tb, ent_key, a_pass in after_results:
            if tb != task_key:
                continue
            counts["total"] += 1
            b_pass = before_pass_by_entity.get(ent_key, False)
            if a_pass and b_pass:
                counts["real"] += 1
            elif a_pass:                        # not b_pass
                counts["trivial"] += 1
            elif b_pass:                        # not a_pass
                counts["knew_failed"] += 1
            else:
                counts["never_knew"] += 1
        trivial_analysis[task_key] = counts

    # =========================================================
    # Score = raw u-pass average over all evaluated questions
    # =========================================================
    after_pass = sum(p for _, _, p in after_results)
    after_total = len(after_results)
    before_pass = sum(v[0] for v in before_per_task.values())
    before_total = sum(v[1] for v in before_per_task.values())
    total = after_total + before_total
    score = (after_pass + before_pass) / total if total else 0.0

    # =========================================================
    # Verbose info string for prompt_model (OfferFeedback)
    # =========================================================
    info_parts = []
    # After raw per-task — ER/Agg/Tr have no trivial concept
    for tb in ("ER", "Agg", "Tr"):
        if tb in after_per_task:
            p, t = after_per_task[tb]
            info_parts.append(f"{tb}:{p}/{t}")
    # After with trivial breakdown for Cas/Abs/Del
    for tb in ("Cas", "Abs", "Del"):
        if tb in after_per_task:
            p, t = after_per_task[tb]
            ta = trivial_analysis[tb]
            info_parts.append(
                f"{tb} after:{p}/{t} "
                f"(real={ta['real']} triv={ta['trivial']} "
                f"kbf={ta['knew_failed']} nk={ta['never_knew']})"
            )
    # Before phase summary
    before_bits = []
    for tb in ("Cas", "Abs", "Del", "Agg", "Tr"):
        if tb in before_per_task:
            p, t = before_per_task[tb]
            before_bits.append(f"{tb}-bef:{p}/{t}")
    if before_bits:
        info_parts.append(" ".join(before_bits))

    info = " | ".join(info_parts) if info_parts else "(no questions evaluated)"

    per_task = {
        "after": {tb: {"pass": v[0], "total": v[1]} for tb, v in after_per_task.items()},
        "before": {tb: {"pass": v[0], "total": v[1]} for tb, v in before_per_task.items()},
    }

    return dspy.Prediction(
        score=score,
        info=info,
        per_task=per_task,
        trivial_analysis=trivial_analysis,
    )
