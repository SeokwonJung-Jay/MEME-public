"""Build flat MEME dataset JSON files from per-stage outputs.

Produces (matches the HuggingFace release at meme-benchmark/MEME):
- meme_nofiller.json    (100 eps, evidence-only sessions)
- meme_filler32k.json   (100 eps, ~32K filler injection)
- meme_filler128k.json  (40 eps subset, ~128K filler injection)

Inputs are the directories produced by the generation pipeline:
- episodes-dir   : Stage 1 (generate_episode.py) output
- nofiller-dir   : Stage 5 (assemble_episodes.py) run with no filler
- filler32k-dir  : Stage 5 run targeting ~32K filler tokens
- filler128k-dir : Stage 5 run targeting ~128K filler tokens (40-ep subset)

Each input directory contains domain-suffixed subfolders:
  episodes_pl/ episodes_sw/, filler_pl/ filler_sw/, nofiller_pl/ nofiller_sw/
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "dataset"

DOMAINS = ("pl", "sw")
DOMAIN_FULL = {"pl": "personal_life", "sw": "software_project"}
TASK_TAGS_NO_HOP = {"ER", "Agg", "Tr", "Del"}
TASK_TAGS_WITH_HOP = {"Cas", "Abs"}


def normalize_task(t):
    """Pass through task `type` and keep `hop` only for Cas/Abs."""
    tag = t["type"]
    if tag in TASK_TAGS_NO_HOP:
        t.pop("hop", None)
    elif tag in TASK_TAGS_WITH_HOP:
        if "hop" not in t:
            raise ValueError(f"{tag} task missing `hop`: {t}")
    else:
        raise ValueError(f"Unknown task type: {tag!r}")
    return t


def normalize_question(q):
    tag = q["task_type"]
    if tag in TASK_TAGS_NO_HOP:
        q.pop("hop", None)
    elif tag in TASK_TAGS_WITH_HOP:
        if "hop" not in q:
            raise ValueError(f"{tag} question missing `hop`: {q}")
    else:
        raise ValueError(f"Unknown task type: {tag!r}")
    return q


def _normalize_q_block(block):
    if isinstance(block, dict):
        return {**block, "questions": [normalize_question(dict(q)) for q in block.get("questions", [])]}
    return block


def load_episode_metadata(episodes_dir, domain, ep_num):
    p = episodes_dir / f"episodes_{domain}" / f"episode_{ep_num:03d}.json"
    return json.load(open(p))


def load_assembled(condition_dir, domain, ep_num, subdir_prefix):
    p = condition_dir / f"{subdir_prefix}_{domain}" / f"episode_{ep_num:03d}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def _build_entry(meta, cond, domain, ep_num):
    tasks = [normalize_task(dict(t)) for t in meta.get("tasks", [])]
    bq = _normalize_q_block(cond.get("before_questions", []))
    aq = _normalize_q_block(cond.get("after_questions", []))
    return {
        "episode_id": f"{domain}_{ep_num:03d}",
        "domain": DOMAIN_FULL[domain],
        "root": meta.get("root"),
        "root_change": meta.get("root_change"),
        "chain_entities": meta.get("chain_entities", []),
        "filler_entities": meta.get("filler_entities", []),
        "entities": meta.get("entities", {}),
        "has_2hop": meta.get("has_2hop", False),
        "dependency_edges_used": meta.get("dependency_edges_used", []),
        "tasks": tasks,
        "total_sessions": cond.get("total_sessions"),
        "evidence_sessions": cond.get("evidence_sessions"),
        "filler_sessions": cond.get("filler_sessions"),
        "total_tokens": cond.get("total_tokens"),
        "evidence_tokens": cond.get("evidence_tokens"),
        "filler_tokens": cond.get("filler_tokens"),
        "evidence_session_indices": cond.get("evidence_session_indices", []),
        "sessions": cond.get("sessions", []),
        "before_questions": bq,
        "after_questions": aq,
    }


def build_condition(name, episodes_dir, condition_dir, subdir_prefix, num_eps_per_domain=50):
    print(f"Building meme_{name}.json")
    entries = []
    for domain in DOMAINS:
        for n in range(1, num_eps_per_domain + 1):
            cond = load_assembled(condition_dir, domain, n, subdir_prefix)
            if cond is None:
                continue
            meta = load_episode_metadata(episodes_dir, domain, n)
            entries.append(_build_entry(meta, cond, domain, n))
    return entries


def write_dataset(name, entries, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"meme_{name}.json"
    json.dump(entries, open(out, "w"), ensure_ascii=False, indent=2)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  wrote {out}: {len(entries)} entries ({size_mb:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--episodes-dir", type=Path, required=True,
                   help="Stage 1 output (contains episodes_pl/, episodes_sw/)")
    p.add_argument("--nofiller-dir", type=Path, required=True,
                   help="Stage 5 nofiller output (contains nofiller_pl/, nofiller_sw/)")
    p.add_argument("--filler32k-dir", type=Path, required=True,
                   help="Stage 5 ~32K-filler output (contains filler_pl/, filler_sw/)")
    p.add_argument("--filler128k-dir", type=Path, required=True,
                   help="Stage 5 ~128K-filler output (40-ep subset; contains filler_pl/, filler_sw/)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help=f"Output directory (default: {DEFAULT_OUT})")
    args = p.parse_args()

    write_dataset("nofiller",
                  build_condition("nofiller", args.episodes_dir, args.nofiller_dir, "nofiller"),
                  args.out_dir)
    write_dataset("filler32k",
                  build_condition("filler32k", args.episodes_dir, args.filler32k_dir, "filler"),
                  args.out_dir)
    write_dataset("filler128k",
                  build_condition("filler128k", args.episodes_dir, args.filler128k_dir, "filler"),
                  args.out_dir)


if __name__ == "__main__":
    main()
