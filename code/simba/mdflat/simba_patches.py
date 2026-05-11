"""Monkey-patches for DSPy SIMBA.

Three patches applied at import time:

1. `append_a_rule` is replaced with a version that logs:
   - entry (good/bad scores, percentile bounds)
   - whether the skip-condition was hit
   - the prompt_model being used
   - the raw OfferFeedback output (discussion + module_advice)
   - the advice applied to each predictor
   Any exception from prompt_model parsing is re-raised (NOT swallowed).

2. `wrap_program` is replaced with a version that does NOT catch program
   exceptions or metric exceptions. Errors propagate, halting the run.

3. The inner `try/except` inside `SIMBA.compile` that wraps `strategy(...)`
   is neutralized by patching the `dspy.teleprompt.simba` module logger so
   that a call matching "Strategy failed with error:" raises instead of
   logging. This causes the except-block's `continue` to never execute; the
   original exception propagates out of compile.

Why do this: silent fallback was hiding that Llama-3.1-8B failed to produce
a parseable OfferFeedback dict on every iteration of SIMBA. Removing the
fallback + logging the LM output lets us verify the failure mode concretely.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

import dspy
import orjson
from dspy.teleprompt import simba as _simba_module
from dspy.teleprompt import simba_utils as _utils_module
from dspy.teleprompt.simba_utils import (
    OfferFeedback,
    inspect_modules,
    recursive_mask,
)


_log = logging.getLogger("simba_patches")


# ============================================================
# 1. append_a_rule — re-implement with full logging, re-raise on failure
# ============================================================

def _traced_append_a_rule(bucket, system, **kwargs):
    """Full-trace replacement for dspy.teleprompt.simba_utils.append_a_rule.

    Logs every step of rule generation. Raises on LM/parser failure (no
    fallback). Produces the same side effects as the original on success.
    """
    predictor2name = kwargs["predictor2name"]
    batch_10p = kwargs["batch_10p_score"]
    batch_90p = kwargs["batch_90p_score"]
    prompt_model = kwargs["prompt_model"] or dspy.settings.lm

    good, bad = bucket[0], bucket[-1]

    _log.info(
        f"[append_a_rule] enter — good_score={good['score']} bad_score={bad['score']} "
        f"batch_10p={batch_10p} batch_90p={batch_90p}"
    )

    if good["score"] <= batch_10p or bad["score"] >= batch_90p:
        _log.info(
            f"[append_a_rule] SKIP — good_score<={batch_10p} OR bad_score>={batch_90p}. "
            f"No advice generated this bucket."
        )
        return False

    if good["score"] <= bad["score"]:
        if good["score"] > batch_90p:
            bad["trace"] = []
            bad["score"] = "N/A"
            bad["prediction"] = {"N/A": "Prediction not available"}
        else:
            good["trace"] = []
            good["score"] = "N/A"
            good["prediction"] = {"N/A": "Prediction not available"}

    module_names = [name for name, _ in system.named_predictors()]
    example = good["example"]

    better_trajectory = [
        {"module_name": predictor2name[id(p)], "inputs": i, "outputs": dict(o)}
        for p, i, o in good["trace"]
    ]
    worse_trajectory = [
        {"module_name": predictor2name[id(p)], "inputs": i, "outputs": dict(o)}
        for p, i, o in bad["trace"]
    ]

    offer_kwargs = {
        "program_code": inspect.getsource(system.__class__),
        "modules_defn": inspect_modules(system),
        "program_inputs": {**example.inputs()},
        "oracle_metadata": {**example.labels()},
        "better_program_trajectory": better_trajectory,
        "better_program_outputs": dict(good["prediction"]),
        "worse_program_trajectory": worse_trajectory,
        "worse_program_outputs": dict(bad["prediction"] or {}),
        "worse_reward_value": bad["score"],
        "better_reward_value": good["score"],
        "worse_reward_info": bad["output_metadata"],
        "better_reward_info": good["output_metadata"],
        "module_names": module_names,
    }
    offer_kwargs = {
        k: v if isinstance(v, str)
        else orjson.dumps(recursive_mask(v), option=orjson.OPT_INDENT_2).decode()
        for k, v in offer_kwargs.items()
    }

    _log.info(
        f"[append_a_rule] calling OfferFeedback via prompt_model="
        f"{getattr(prompt_model, 'model', type(prompt_model).__name__)} — "
        f"modules={module_names}, "
        f"program_inputs_len={len(offer_kwargs['program_inputs'])}, "
        f"better_traj_len={len(offer_kwargs['better_program_trajectory'])}, "
        f"worse_traj_len={len(offer_kwargs['worse_program_trajectory'])}"
    )

    # NO try/except — a parse failure here propagates. This is intentional:
    # we want to SEE when the prompt_model fails to produce valid output.
    with dspy.context(trace=[], lm=prompt_model):
        advice_program = dspy.Predict(OfferFeedback)
        result = advice_program(**offer_kwargs)

    advice = result.module_advice
    discussion = getattr(result, "discussion", "")

    _log.info(
        f"[append_a_rule] OfferFeedback returned — "
        f"discussion_len={len(discussion)} advice_type={type(advice).__name__} "
        f"advice_keys={list(advice.keys()) if isinstance(advice, dict) else 'N/A'}"
    )
    _log.info(f"[append_a_rule] discussion (first 500 chars): {discussion[:500]}")

    applied = 0
    for name, predictor in system.named_predictors():
        if name in advice:
            _log.info(f"[append_a_rule] Advice for {name} (first 400 chars): {advice[name][:400]}")
            instructions = predictor.signature.instructions + "\n\n" + advice[name]
            predictor.signature = predictor.signature.with_instructions(instructions)
            applied += 1
        else:
            _log.warning(f"[append_a_rule] NO advice for module '{name}' — it was missing from the dict")

    _log.info(f"[append_a_rule] applied advice to {applied}/{len(module_names)} modules")
    return True


_utils_module.append_a_rule = _traced_append_a_rule
# simba.py imports it directly: `from ...simba_utils import ... append_a_rule`
# so we also need to patch the name there
if hasattr(_simba_module, "append_a_rule"):
    _simba_module.append_a_rule = _traced_append_a_rule


# ============================================================
# 2. wrap_program — no exception catching
# ============================================================

def _non_catching_wrap_program(program, metric: Callable):
    def wrapped(example):
        with dspy.context(trace=[]):
            prediction = program(**example.inputs())  # raises on failure
            trace = dspy.settings.trace.copy()

        output = metric(example, prediction)
        output_metadata: dict[str, Any] = {}
        if isinstance(output, (int, float)):
            score = output
        elif isinstance(output, dspy.Prediction):
            if not hasattr(output, "score"):
                raise ValueError("metric's Prediction must contain a 'score' field")
            score = output.score
            output_metadata = {k: v for k, v in output.items() if k != "score"}
        else:
            raise TypeError(f"metric returned unsupported type: {type(output).__name__}")

        return {
            "prediction": prediction,
            "trace": trace,
            "score": score,
            "example": example,
            "output_metadata": output_metadata,
        }
    return wrapped


_utils_module.wrap_program = _non_catching_wrap_program
if hasattr(_simba_module, "wrap_program"):
    _simba_module.wrap_program = _non_catching_wrap_program


# ============================================================
# 3. Neutralize SIMBA.compile's strategy try/except
# ============================================================
#
# SIMBA.compile has this block (dspy/teleprompt/simba.py:282-294):
#   try:
#       strategy(bucket, system_candidate, ...)
#   except Exception as e:
#       logger.error(f"Strategy failed with error: {e}")
#       continue
#
# We can't surgically edit the method without copying the whole body. But we
# can hijack `logger.error`: if the message starts with "Strategy failed",
# raise. Raising inside an except-handler propagates up, past `continue`.

_simba_logger = logging.getLogger("dspy.teleprompt.simba")
_original_error = _simba_logger.error


def _strict_error(msg, *args, **kwargs):
    # Let the original log line be emitted first so it's visible.
    _original_error(msg, *args, **kwargs)
    if isinstance(msg, str) and "Strategy failed with error" in msg:
        raise RuntimeError(f"SIMBA strategy failed (fallback disabled): {msg}")


_simba_logger.error = _strict_error


# ============================================================
# 4. append_a_demo — wrap factory to log demo-adding strategy events
# ============================================================
#
# SIMBA.strategies contains [append_a_demo(maxlen), append_a_rule]. Without a
# patch, `append_a_demo` invocations are invisible in our logs (they don't
# modify `.signature.instructions`, only `.demos`), so monitoring only
# `append_a_rule] enter` misses 50% of SIMBA's work.

_original_append_a_demo = _utils_module.append_a_demo


def _traced_append_a_demo(demo_input_field_maxlen):
    inner = _original_append_a_demo(demo_input_field_maxlen)

    def _wrapper(bucket, system, **kwargs):
        good = bucket[0]
        batch_10p = kwargs.get("batch_10p_score")
        _log.info(
            f"[append_a_demo] enter — good_score={good['score']} "
            f"batch_10p={batch_10p}"
        )
        try:
            result = inner(bucket, system, **kwargs)
        except Exception as e:
            _log.error(f"[append_a_demo] FAILED: {type(e).__name__}: {e}")
            raise
        _log.info(f"[append_a_demo] result={result}")
        if result:
            for name, predictor in system.named_predictors():
                _log.info(
                    f"[append_a_demo] {name} now has {len(predictor.demos)} demos"
                )
        return result

    return _wrapper


_utils_module.append_a_demo = _traced_append_a_demo
if hasattr(_simba_module, "append_a_demo"):
    _simba_module.append_a_demo = _traced_append_a_demo


_log.info(
    "simba_patches: installed (append_a_rule trace + wrap_program strict + "
    "strategy-no-continue + append_a_demo trace)"
)
