"""In-context baseline runner — Table 2 top rows.

Bypasses every memory system and feeds the full episode transcript
(all sessions concatenated) directly to the answering LLM as context,
then asks the question. This is distinct from `golden_memory.py`,
which feeds only the gold facts to measure the upper-bound ceiling.

Usage:
  python3 in_context_baseline.py -d ../data/filler32k_pl --model gpt-4.1-mini -o output/in_context/gpt41mini
  python3 in_context_baseline.py -d ../data/filler32k_pl --model claude-sonnet-4-6 -o output/in_context/sonnet46
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI

from eval.budget_tracker import install_patches, get_tracker

install_patches()


SYSTEM_PROMPT = (
    "You are answering a question based on a long conversation transcript. "
    "Read the transcript carefully and answer the question using only information present "
    "in it. Be concise and answer with the value only."
)


def _make_client(model: str, api_key: str):
    if model.startswith("claude"):
        from agents.anthropic_adapter import AnthropicAsOpenAI
        return AnthropicAsOpenAI(api_key=api_key)
    return OpenAI(api_key=api_key)


def _flatten_sessions(sessions, up_to: int) -> str:
    parts = []
    for i in range(min(up_to, len(sessions))):
        sess = sessions[i]
        ts = sess.get("timestamp", f"session_{i}")
        parts.append(f"[Session {ts}]")
        for turn in sess.get("conversation", []):
            role = "User" if turn.get("role") == "user" else "Assistant"
            parts.append(f"{role}: {turn.get('content','')}")
    return "\n".join(parts)


def _ask(client, model: str, transcript: str, question: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n{transcript}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=500,
    )
    msg = response.choices[0].message
    return (msg.content or "").strip()


def process_one_episode(ep_path: str, model: str, api_key: str, output_dir: str):
    with open(ep_path) as f:
        episode = json.load(f)

    ep_id = episode["episode_id"]
    domain = episode["domain"]
    domain_prefix = {"personal_life": "pl", "software_project": "sw"}[domain]
    sessions = episode["sessions"]

    before_pos = episode["before_questions"]["position_after_session"] + 1
    after_pos = episode["after_questions"]["position_after_session"] + 1

    transcript_before = _flatten_sessions(sessions, before_pos)
    transcript_after = _flatten_sessions(sessions, after_pos)

    client = _make_client(model, api_key)
    tracker = get_tracker()
    tracker.reset()

    print(f"  Ep{ep_id:3d} START ({domain_prefix})")
    t0 = time.time()

    before_answers = []
    for q in episode["before_questions"]["questions"]:
        ans = _ask(client, model, transcript_before, q["question"])
        before_answers.append({**q, "agent_answer": ans, "retrieved_context": "(full transcript)"})

    after_answers = []
    for q in episode["after_questions"]["questions"]:
        ans = _ask(client, model, transcript_after, q["question"])
        after_answers.append({**q, "agent_answer": ans, "retrieved_context": "(full transcript)"})

    output = {
        "episode_id": ep_id,
        "domain": domain,
        "before_answers": before_answers,
        "after_answers": after_answers,
        "config": {
            "agent_type": "in_context",
            "agent_model": model,
            "internal_model": None,
        },
        "budget": tracker.snapshot(),
    }

    os.makedirs(output_dir, exist_ok=True)
    model_tag = model.replace("/", "-")
    out_path = os.path.join(
        output_dir,
        f"agent_{domain_prefix}_{ep_id:03d}_in_context_{model_tag}.json",
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Ep{ep_id:3d} DONE  ({time.time()-t0:.0f}s) → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="In-context baseline (Table 2 top rows)")
    parser.add_argument("-d", "--episode_dir", required=True,
                        help="Directory of unpacked episodes (e.g., data/filler32k_pl)")
    parser.add_argument("-o", "--output_dir", default="output/in_context",
                        help="Where to write per-episode JSON outputs")
    parser.add_argument("--model", default="gpt-4.1-mini",
                        help="Answering LLM (gpt-4.1-mini, claude-sonnet-4-6, ...)")
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.model.startswith("claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set"); sys.exit(1)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not set"); sys.exit(1)

    episode_files = sorted(
        os.path.join(args.episode_dir, f)
        for f in os.listdir(args.episode_dir)
        if f.startswith("episode_") and f.endswith(".json")
    )
    if not episode_files:
        print("No episode files found."); sys.exit(1)

    if args.skip_existing:
        model_tag = args.model.replace("/", "-")
        kept = []
        for ep_path in episode_files:
            with open(ep_path) as f:
                _ep = json.load(f)
            _prefix = {"personal_life": "pl", "software_project": "sw"}[_ep["domain"]]
            _out = os.path.join(args.output_dir,
                f"agent_{_prefix}_{_ep['episode_id']:03d}_in_context_{model_tag}.json")
            if not os.path.exists(_out):
                kept.append(ep_path)
        episode_files = kept

    print(f"Running in-context baseline on {len(episode_files)} episodes (model={args.model})")

    if args.workers == 1:
        for ep_path in episode_files:
            process_one_episode(ep_path, args.model, api_key, args.output_dir)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one_episode, ep, args.model, api_key, args.output_dir)
                    for ep in episode_files]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
