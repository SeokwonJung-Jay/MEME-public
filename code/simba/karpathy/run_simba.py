"""Run SIMBA prompt optimization on Karpathy Wiki memory system.

Optimizes 3 prompts (flush / compile / query) using DSPy SIMBA.
All 3 phases run on gpt-4.1-mini (Karpathy's native-LLM convention;
query output IS the answer, so no separate answer LM).

Prompt generator LM: gpt-4.1-mini by default (overridable with --prompt-model).
Original .deps/claude-memory-compiler/ is NOT modified — prompts are overridden
at runtime via KarpathyProgram (DSPy Module).

Usage:
  python run_simba.py --train 10 --test 10 --seed 7 --max-steps 2
"""
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
_HERE = Path(__file__).resolve().parent
for _env in (_HERE / ".env", _HERE.parent.parent / ".env"):
    if _env.exists():
        load_dotenv(_env, override=False)
        break

import dspy
import simba_patches  # noqa: F401 — installs SIMBA monkey-patches (traced append_a_rule, no-catch wrap_program, no-continue strategy)
from dspy.teleprompt import SIMBA

from data_loader import load_trainset_testset
from karpathy_module import KarpathyProgram
from metric import u_check_score


def parallel_eval(program, examples, label, num_threads=4):
    results = [None] * len(examples)

    def _one(i, ex):
        r = u_check_score(ex, program(**ex.inputs()))
        score = r.score if hasattr(r, 'score') else r
        info = r.info if hasattr(r, 'info') else ""
        return i, score, info

    with ThreadPoolExecutor(max_workers=num_threads) as tp:
        futures = [tp.submit(_one, i, examples[i]) for i in range(len(examples))]
        for fut in as_completed(futures):
            i, score, info = fut.result()
            results[i] = (score, info)
            logging.info(f"  {label} ep{i+1}/{len(examples)}: score={score:.3f} | {info}")
    return results


def setup_logging(output_dir: Path):
    log_path = output_dir / "run.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    # Ensure DSPy SIMBA loggers propagate (Advice for X / batch / strategy / candidate scores)
    logging.getLogger("dspy").setLevel(logging.INFO)
    logging.getLogger("dspy.teleprompt.simba").setLevel(logging.INFO)
    logging.getLogger("dspy.teleprompt.simba_utils").setLevel(logging.INFO)
    logging.info(f"=== Run started at {datetime.now().isoformat()} ===")
    logging.info(f"Logging to {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=10)
    parser.add_argument("--test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--bsize", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2,
                        help="SIMBA max_steps (total iterations). 1 = single-batch sanity run.")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--task-model", default="openai/gpt-4.1-mini",
                        help="DSPy LM prefix — used by SIMBA program LM (not for our tool loops)")
    parser.add_argument("--internal-model", default="gpt-4.1-mini",
                        help="Model used inside flush/compile/query tool loops. Must be an "
                             "OpenAI-compatible model (the tool loop uses the OpenAI SDK directly).")
    parser.add_argument("--prompt-model", default="openai/gpt-4.1-mini",
                        help="LM that generates prompt candidates (OfferFeedback).")
    parser.add_argument("--api-base", default=None,
                        help="Optional base URL (only for hosted_vllm/ prompt-model).")

    _here = Path(__file__).resolve().parent
    _default_data = None
    for _cand in [_here / "data",
                  _here.parent.parent / "data"]:
        if (_cand / "filler32k_pl").exists():
            _default_data = _cand
            break
    parser.add_argument("--data-dir", type=Path, default=_default_data)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    logging.info(f"Run dir: {run_dir}")
    logging.info(f"Args: {vars(args)}")

    # Enable determinism aids + per-tool-call JSONL logging. openai_tool_agent
    # and karpathy_module._run_flush_llm both read these env vars; main
    # benchmark doesn't set them, so behavior outside SIMBA is unchanged.
    os.environ.setdefault("OPENAI_SEED", str(args.seed))
    tool_log_path = run_dir / "tool_calls.jsonl"
    os.environ["TOOL_CALL_LOG_PATH"] = str(tool_log_path)
    logging.info(f"OPENAI_SEED={os.environ['OPENAI_SEED']}, tool-call log: {tool_log_path}")

    # Configure DSPy LMs — SIMBA needs a default task_lm to attach to the program
    # (even though our tool loops bypass it). Use task_model for this.
    task_lm = dspy.LM(args.task_model, temperature=0, cache=False)
    prompt_kwargs = {"temperature": 0.9, "cache": False}
    if args.prompt_model.startswith("hosted_vllm/"):
        if not args.api_base:
            raise RuntimeError("--api-base required for hosted_vllm/ prompt-model")
        prompt_kwargs["api_base"] = args.api_base
        prompt_kwargs["api_key"] = "EMPTY"
    prompt_lm = dspy.LM(args.prompt_model, **prompt_kwargs)
    dspy.configure(lm=task_lm)

    if args.data_dir is None or not (args.data_dir / "filler32k_pl").exists():
        raise RuntimeError(f"data_dir invalid: {args.data_dir}")
    logging.info(f"Using data_dir: {args.data_dir}")

    n_train_per_domain = args.train // 2
    n_test_per_domain = args.test // 2
    trainset, testset = load_trainset_testset(
        args.data_dir,
        n_train_per_domain=n_train_per_domain,
        n_test_per_domain=n_test_per_domain,
        seed=args.seed,
    )
    logging.info(f"Train: {len(trainset)} episodes")
    for ex in trainset:
        logging.info(f"  train: {ex.domain} ep{ex.episode_id}")
    logging.info(f"Test:  {len(testset)} episodes")
    for ex in testset:
        logging.info(f"  test: {ex.domain} ep{ex.episode_id}")

    program = KarpathyProgram(model=args.internal_model)

    logging.info("=== Baseline eval on trainset ===")
    baseline_train_results = parallel_eval(program, trainset, "baseline train", args.num_threads)
    baseline_train_scores = [s for s, _ in baseline_train_results]
    baseline_train_info = [i for _, i in baseline_train_results]
    baseline_train = sum(baseline_train_scores) / len(baseline_train_scores)
    logging.info(f"Baseline train avg: {baseline_train:.3f}")

    logging.info("=== Baseline eval on testset ===")
    baseline_test_results = parallel_eval(program, testset, "baseline test", args.num_threads)
    baseline_test_scores = [s for s, _ in baseline_test_results]
    baseline_test_info = [i for _, i in baseline_test_results]
    baseline_test = sum(baseline_test_scores) / len(baseline_test_scores)
    logging.info(f"Baseline test avg: {baseline_test:.3f}")

    with open(run_dir / "baseline_report.json", "w") as f:
        json.dump({
            "baseline_train": baseline_train,
            "baseline_test": baseline_test,
            "baseline_train_per_episode": baseline_train_scores,
            "baseline_test_per_episode": baseline_test_scores,
        }, f, indent=2)
    logging.info(f"Saved baseline_report.json")

    logging.info(f"=== SIMBA optimization (num_candidates={args.num_candidates}, "
                 f"bsize={args.bsize}, max_steps={args.max_steps}) ===")
    optimizer = SIMBA(
        metric=u_check_score,
        num_candidates=args.num_candidates,
        bsize=args.bsize,
        max_steps=args.max_steps,
        num_threads=args.num_threads,
        prompt_model=prompt_lm,
    )
    optimized = optimizer.compile(program, trainset=trainset)
    logging.info("SIMBA complete.")

    prompts = {
        "flush": optimized.flush.signature.instructions,
        "compile": optimized.compile.signature.instructions,
        "query": optimized.query.signature.instructions,
    }
    with open(run_dir / "optimized_prompts.json", "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved optimized_prompts.json")

    optimized.save(str(run_dir / "optimized_program.json"))
    logging.info(f"Saved optimized_program.json")

    # Full candidate audit trail: per-head prompts + train_score per rank
    candidate_dumps = []
    for i, c in enumerate(getattr(optimized, "candidate_programs", []) or []):
        prog = c["program"]
        candidate_dumps.append({
            "rank": i,
            "train_score": c["score"],
            "prompts": {
                "flush": prog.flush.signature.instructions,
                "compile": prog.compile.signature.instructions,
                "query": prog.query.signature.instructions,
            },
        })
    with open(run_dir / "all_candidates.json", "w") as f:
        json.dump(candidate_dumps, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved all_candidates.json ({len(candidate_dumps)} candidates)")

    logging.info("=== Optimized eval on trainset ===")
    opt_train_results = parallel_eval(optimized, trainset, "opt train", args.num_threads)
    opt_train_scores = [s for s, _ in opt_train_results]
    opt_train_info = [i for _, i in opt_train_results]
    opt_train = sum(opt_train_scores) / len(opt_train_scores)
    logging.info(f"Optimized train avg: {opt_train:.3f}")

    logging.info("=== Optimized eval on testset ===")
    opt_test_results = parallel_eval(optimized, testset, "opt test", args.num_threads)
    opt_test_scores = [s for s, _ in opt_test_results]
    opt_test_info = [i for _, i in opt_test_results]
    opt_test = sum(opt_test_scores) / len(opt_test_scores)
    logging.info(f"Optimized test avg: {opt_test:.3f}")

    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "run_dir": str(run_dir),
        "trainset_size": len(trainset),
        "testset_size": len(testset),
        "train_episodes": [f"{ex.domain}_{ex.episode_id}" for ex in trainset],
        "test_episodes": [f"{ex.domain}_{ex.episode_id}" for ex in testset],
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "optimized_train": opt_train,
        "optimized_test": opt_test,
        "improvement_train_pp": (opt_train - baseline_train) * 100,
        "improvement_test_pp": (opt_test - baseline_test) * 100,
        "baseline_train_per_episode": baseline_train_scores,
        "baseline_test_per_episode": baseline_test_scores,
        "optimized_train_per_episode": opt_train_scores,
        "optimized_test_per_episode": opt_test_scores,
        "baseline_train_info_per_episode": baseline_train_info,
        "baseline_test_info_per_episode": baseline_test_info,
        "optimized_train_info_per_episode": opt_train_info,
        "optimized_test_info_per_episode": opt_test_info,
        "prompts": prompts,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved report.json")

    logging.info("=" * 60)
    logging.info("FINAL SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Baseline train: {baseline_train:.3f}  →  Optimized: {opt_train:.3f}  (+{(opt_train-baseline_train)*100:+.1f}pp)")
    logging.info(f"Baseline test:  {baseline_test:.3f}  →  Optimized: {opt_test:.3f}  (+{(opt_test-baseline_test)*100:+.1f}pp)")
    logging.info(f"All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
