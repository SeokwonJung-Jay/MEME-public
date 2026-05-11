"""Run SIMBA prompt optimization on MD-flat.

Optimizes 3 prompts (ingest / retrieve / answer) using DSPy SIMBA.
Task LM: gpt-4.1-mini (internal model, same as our experiments).
Prompt generator LM: Llama-3.1-8B-Instruct via HF Inference.

Usage:
  python run_simba.py --train 10 --test 30 --seed 42 --rounds 3
"""
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Load .env (checks experiment dir first, then repo root) BEFORE importing dspy/openai
from dotenv import load_dotenv
_HERE = Path(__file__).resolve().parent
for _env in (_HERE / ".env", _HERE.parent.parent / ".env"):
    if _env.exists():
        load_dotenv(_env, override=False)
        break

import dspy
import simba_patches  # noqa: F401 — installs monkey-patches at import time
from dspy.teleprompt import SIMBA


def parallel_eval(program, examples, label, num_threads=4):
    """Run u_check_score on examples in parallel. Returns list of (score, info) tuples."""
    from metric import u_check_score
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

from data_loader import load_trainset_testset
from mdflat_module import MDFlatProgram
from metric import u_check_score


def setup_logging(output_dir: Path):
    """Dual logging: stdout + file. Timestamped lines."""
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
    # Ensure DSPy SIMBA loggers propagate to our handlers (Advice/batch/strategy logs)
    logging.getLogger("dspy").setLevel(logging.INFO)
    logging.getLogger("dspy.teleprompt.simba").setLevel(logging.INFO)
    logging.getLogger("dspy.teleprompt.simba_utils").setLevel(logging.INFO)
    logging.info(f"=== Run started at {datetime.now().isoformat()} ===")
    logging.info(f"Logging to {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=10,
                        help="Train episodes (split evenly PL/SW)")
    parser.add_argument("--test", type=int, default=10,
                        help="Test episodes (split evenly PL/SW)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-candidates", type=int, default=4,
                        help="SIMBA num_candidates")
    parser.add_argument("--bsize", type=int, default=4,
                        help="SIMBA batch size")
    parser.add_argument("--max-steps", type=int, default=2,
                        help="SIMBA max_steps (total iterations). 1 = single-batch sanity run.")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="SIMBA parallel evaluation threads")
    parser.add_argument("--task-model", default="openai/gpt-4.1-mini",
                        help="Internal LM for ingest+retrieve tool loops. Must be an "
                             "OpenAI-compatible model (the tool loop uses the OpenAI SDK directly).")
    parser.add_argument("--answer-model", default="anthropic/claude-sonnet-4-20250514",
                        help="Answer LM (final answer phase, matches production setup)")
    parser.add_argument("--prompt-model", default="openai/gpt-4.1-mini",
                        help="LM that generates prompt candidates (OfferFeedback). Default: gpt-4.1-mini.")
    parser.add_argument("--api-base", default=None,
                        help="Optional base URL (only needed for hosted_vllm/... style prompt-model).")
    # Auto-find data dir under the standard release layout (code/data/)
    _here = Path(__file__).resolve().parent
    _default_data = None
    for _cand in [_here / "data",
                  _here.parent.parent / "data"]:
        if (_cand / "filler32k_pl").exists():
            _default_data = _cand
            break
    parser.add_argument("--data-dir", type=Path, default=_default_data,
                        help="Path containing filler32k_pl/ and filler32k_sw/")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--run-name", default=None,
                        help="Subfolder name under output-dir (default: timestamp)")
    args = parser.parse_args()

    # Per-run subfolder so multiple runs don't overwrite
    if args.run_name is None:
        args.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(run_dir)
    logging.info(f"Run dir: {run_dir}")
    logging.info(f"Args: {vars(args)}")

    # Enable per-tool-call JSONL logging (mdflat_module reads this env var).
    # Captures every gpt-4.1-mini tool call during baseline eval, SIMBA
    # optimization, and optimized eval so we can diff runs post-hoc.
    tool_log_path = run_dir / "tool_calls.jsonl"
    os.environ["TOOL_CALL_LOG_PATH"] = str(tool_log_path)
    logging.info(f"Tool-call log: {tool_log_path}")

    # Configure DSPy LMs
    task_lm = dspy.LM(args.task_model, temperature=0, cache=False)
    answer_lm = dspy.LM(args.answer_model, temperature=0, cache=False, max_tokens=500)

    # prompt_model: pass api_base only if explicitly needed (e.g. hosted_vllm/)
    prompt_kwargs = {"temperature": 0.9, "cache": False}
    if args.prompt_model.startswith("hosted_vllm/"):
        if not args.api_base:
            raise RuntimeError("--api-base required for hosted_vllm/ prompt-model")
        prompt_kwargs["api_base"] = args.api_base
        prompt_kwargs["api_key"] = "EMPTY"
    prompt_lm = dspy.LM(args.prompt_model, **prompt_kwargs)

    dspy.configure(lm=task_lm)

    # Sanity check data dir
    if args.data_dir is None or not (args.data_dir / "filler32k_pl").exists():
        raise RuntimeError(f"data_dir invalid: {args.data_dir}. Pass --data-dir or ensure filler32k_pl/filler32k_sw exist.")
    logging.info(f"Using data_dir: {args.data_dir}")

    # Load data
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

    # Baseline program
    # Extract model names from DSPy-style prefixes ("openai/gpt-4.1-mini" → "gpt-4.1-mini")
    task_model_name = args.task_model.split("/", 1)[-1] if "/" in args.task_model else args.task_model
    answer_model_name = args.answer_model.split("/", 1)[-1] if "/" in args.answer_model else args.answer_model
    program = MDFlatProgram(answer_lm=answer_lm, task_model=task_model_name, answer_model=answer_model_name)

    # Baseline eval on trainset + testset. Wrap both so any exception from a
    # worker thread (timeout, network, unhandled error) writes FAILED.json
    # before re-raising. JSON-parse errors from malformed tool_calls are
    # tolerated inside _run_tool_loop and never reach here.
    try:
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
    except Exception as e:
        logging.exception(f"Baseline eval aborted: {type(e).__name__}: {e}")
        with open(run_dir / "FAILED.json", "w") as f:
            json.dump({
                "stage": "baseline_eval",
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved FAILED.json to {run_dir}")
        raise

    # Save baseline report early (in case SIMBA crashes)
    with open(run_dir / "baseline_report.json", "w") as f:
        json.dump({
            "baseline_train": baseline_train,
            "baseline_test": baseline_test,
            "baseline_train_per_episode": baseline_train_scores,
            "baseline_test_per_episode": baseline_test_scores,
        }, f, indent=2)
    logging.info(f"Saved baseline_report.json")

    # SIMBA optimization — wrapped so a timeout/hang inside any worker thread
    # aborts the whole run (intentional fail-loud). Partial progress (baseline
    # reports + simba_patches' per-candidate log lines) is already on disk.
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
    try:
        optimized = optimizer.compile(program, trainset=trainset)
    except Exception as e:
        logging.exception(f"SIMBA compile aborted: {type(e).__name__}: {e}")
        with open(run_dir / "FAILED.json", "w") as f:
            json.dump({
                "stage": "simba_compile",
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "baseline_train": baseline_train,
                "baseline_test": baseline_test,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved FAILED.json to {run_dir}")
        raise  # fail the whole run, non-zero exit
    logging.info("SIMBA complete.")

    # Save optimized prompts (answer is frozen — excluded from optimization scope)
    prompts = {
        "ingest": optimized.ingest.signature.instructions,
        "retrieve": optimized.retrieve.signature.instructions,
    }
    with open(run_dir / "optimized_prompts.json", "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved optimized_prompts.json")

    # Save serialized program
    optimized.save(str(run_dir / "optimized_program.json"))
    logging.info(f"Saved optimized_program.json")

    # Dump every SIMBA candidate (prompt text + validation score) for full audit trail
    candidate_dumps = []
    for i, c in enumerate(getattr(optimized, "candidate_programs", []) or []):
        prog = c["program"]
        candidate_dumps.append({
            "rank": i,
            "train_score": c["score"],
            "prompts": {
                "ingest": prog.ingest.signature.instructions,
                "retrieve": prog.retrieve.signature.instructions,
            },
        })
    with open(run_dir / "all_candidates.json", "w") as f:
        json.dump(candidate_dumps, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved all_candidates.json ({len(candidate_dumps)} candidates with per-head prompts + train_score)")

    # Optimized eval on train (parallel)
    logging.info("=== Optimized eval on trainset ===")
    opt_train_results = parallel_eval(optimized, trainset, "opt train", args.num_threads)
    opt_train_scores = [s for s, _ in opt_train_results]
    opt_train_info = [i for _, i in opt_train_results]
    opt_train = sum(opt_train_scores) / len(opt_train_scores)
    logging.info(f"Optimized train avg: {opt_train:.3f}")

    # Optimized eval on test (parallel)
    logging.info("=== Optimized eval on testset ===")
    opt_test_results = parallel_eval(optimized, testset, "opt test", args.num_threads)
    opt_test_scores = [s for s, _ in opt_test_results]
    opt_test_info = [i for _, i in opt_test_results]
    opt_test = sum(opt_test_scores) / len(opt_test_scores)
    logging.info(f"Optimized test avg: {opt_test:.3f}")

    # Final report
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

    # Summary print
    logging.info("=" * 60)
    logging.info("FINAL SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Baseline train: {baseline_train:.3f}  →  Optimized: {opt_train:.3f}  (+{(opt_train-baseline_train)*100:+.1f}pp)")
    logging.info(f"Baseline test:  {baseline_test:.3f}  →  Optimized: {opt_test:.3f}  (+{(opt_test-baseline_test)*100:+.1f}pp)")
    logging.info(f"All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
