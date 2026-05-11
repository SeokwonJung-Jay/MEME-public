"""Re-run the ANSWER LM on existing agent output JSONs with updated prompt.

Reuses the stored `retrieved_context` from each question (no re-ingest, no
re-retrieve). Replaces `agent_answer` (and `answer_time_sec`) in-place.

Skips Karpathy (its answer_question is overridden to return query.py output
directly, not subject to UNIFIED_ANSWER_PROMPT).

Usage:
  python reanswer.py -d output/md_file/agent --model claude-sonnet-4-20250514 -w 4

Output: writes to `-o` dir (default: {-d}_reanswered). Original JSON preserved.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
_REPO = Path(__file__).resolve().parent.parent
for _env in (_REPO / ".env",):
    if _env.exists():
        load_dotenv(_env, override=False)
        break

# Import the updated prompt from base
sys.path.insert(0, str(_REPO))
from agents.base import UNIFIED_ANSWER_PROMPT


def _make_client(model: str):
    if model.startswith("claude"):
        from agents.anthropic_adapter import AnthropicAsOpenAI
        return AnthropicAsOpenAI(api_key=os.environ["ANTHROPIC_API_KEY"])
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def reanswer_episode(fp: Path, out_fp: Path, client, model: str, max_workers: int = 8):
    """Re-answer all questions in one agent JSON. Parallelizes across Qs."""
    d = json.load(open(fp))
    t0 = time.time()

    qs_to_redo = []
    for phase in ("before_answers", "after_answers"):
        for i, a in enumerate(d.get(phase, [])):
            qs_to_redo.append((phase, i, a["question"], a.get("retrieved_context", "")))

    def _one(phase, i, question, ctx):
        prompt = UNIFIED_ANSWER_PROMPT.format(question=question, context=ctx)
        t_call = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        elapsed = time.time() - t_call
        ans = resp.choices[0].message.content.strip() if resp.choices[0].message.content else ""
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        return phase, i, ans, in_tok, out_tok, elapsed

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, p, i, q, c): (p, i) for p, i, q, c in qs_to_redo}
        for fut in as_completed(futs):
            results.append(fut.result())

    total_in = total_out = 0
    for phase, i, ans, in_tok, out_tok, elapsed in results:
        d[phase][i]["agent_answer"] = ans
        d[phase][i]["answer_time_sec"] = round(elapsed, 2)
        total_in += in_tok
        total_out += out_tok

    # Preserve budget.answer from the ORIGINAL run (captured via budget_tracker
    # which monkey-patches anthropic.Messages.create — still valid for v2
    # because the prompt change doesn't materially shift per-call token counts).
    # The reanswer-only token counts reported here come from resp.usage which
    # the anthropic_adapter does not currently expose → values may be 0.
    d["reanswer"] = {
        "model": model,
        "prompt_version": "v2_question_first",
        "elapsed_sec": round(time.time() - t0, 2),
        "input_tokens_reported": total_in,   # 0 until adapter .usage is wired
        "output_tokens_reported": total_out,
        "n_questions": len(qs_to_redo),
    }

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

    return {
        "ep_id": d.get("episode_id"),
        "n_qs": len(qs_to_redo),
        "elapsed": round(time.time() - t0, 2),
        "in_tok": total_in,
        "out_tok": total_out,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--agent_dir", required=True, help="Dir of agent_*.json files")
    p.add_argument("-o", "--output_dir", default=None,
                   help="Output dir (default: {agent_dir}_reanswered)")
    p.add_argument("--model", default="claude-sonnet-4-20250514")
    p.add_argument("-w", "--workers", type=int, default=4,
                   help="Parallel EPISODES (each episode uses 8 inner threads for Qs)")
    p.add_argument("--inner-workers", type=int, default=8,
                   help="Parallel Qs within one episode")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip files that already exist in output_dir")
    args = p.parse_args()

    agent_dir = Path(args.agent_dir)
    if args.output_dir is None:
        args.output_dir = str(agent_dir.parent / (agent_dir.name + "_reanswered"))
    out_dir = Path(args.output_dir)

    files = sorted(agent_dir.glob("agent_*.json"))
    if not files:
        print(f"No agent_*.json in {agent_dir}")
        sys.exit(1)

    if args.skip_existing:
        kept = [f for f in files if not (out_dir / f.name).exists()]
        skipped = len(files) - len(kept)
        print(f"--skip-existing: {skipped} done, {len(kept)} remaining")
        files = kept
        if not files:
            print("Nothing to do.")
            sys.exit(0)

    print(f"Re-answering {len(files)} files from {agent_dir}")
    print(f"Output: {out_dir}")
    print(f"Model: {args.model}, episode-parallel={args.workers}, Q-inner={args.inner_workers}")

    t_start = time.time()
    total_in = total_out = 0

    def _ep(fp):
        client = _make_client(args.model)  # one client per thread
        return reanswer_episode(fp, out_dir / fp.name, client, args.model, args.inner_workers)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_ep, fp): fp.name for fp in files}
        done = 0
        for fut in as_completed(futs):
            name = futs[fut]
            r = fut.result()
            done += 1
            total_in += r["in_tok"]
            total_out += r["out_tok"]
            print(f"  [{done}/{len(files)}] {name}  ep{r['ep_id']:3d}  "
                  f"{r['n_qs']}q  {r['elapsed']:.1f}s  "
                  f"in={r['in_tok']} out={r['out_tok']}")

    elapsed = time.time() - t_start
    # Sonnet 4 pricing
    PRICE_IN = 3.0 / 1_000_000
    PRICE_OUT = 15.0 / 1_000_000
    cost = total_in * PRICE_IN + total_out * PRICE_OUT
    print(f"\nDone: {len(files)} files in {elapsed:.0f}s")
    print(f"  tokens: in={total_in:,} out={total_out:,}")
    print(f"  cost: ${cost:.2f}")


if __name__ == "__main__":
    main()
