"""Expand a flat MEME dataset JSON into per-episode files for run_agent.py.

The HF release ships a flat list of episodes; the eval runner reads
per-file episode JSONs split by domain. This helper bridges the two.

Usage:
  python3 unpack_dataset.py --input ../../dataset/meme_filler32k.json --output ../data
  # writes ../data/filler32k_pl/episode_NNN.json and ../data/filler32k_sw/episode_NNN.json

  python3 eval/run_agent.py -d data/filler32k_pl --agent-type bm25
"""

import argparse
import json
import re
from pathlib import Path

_FNAME_RE = re.compile(r"^meme_(?P<condition>[a-z0-9_]+)\.json$")
_EP_ID_RE = re.compile(r"^(?P<domain>pl|sw)_(?P<num>\d{3})$")


def unpack(input_path: Path, output_root: Path):
    m = _FNAME_RE.match(input_path.name)
    if not m:
        raise ValueError(f"Expected filename like 'meme_<condition>.json', got: {input_path.name}")
    condition = m.group("condition")

    episodes = json.load(open(input_path))
    counts = {"pl": 0, "sw": 0}

    for ep in episodes:
        em = _EP_ID_RE.match(str(ep["episode_id"]))
        if not em:
            raise ValueError(f"Unexpected episode_id: {ep['episode_id']!r}")
        domain, num = em.group("domain"), int(em.group("num"))

        out_dir = output_root / f"{condition}_{domain}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ep_out = dict(ep)
        ep_out["episode_id"] = num
        with open(out_dir / f"episode_{num:03d}.json", "w") as f:
            json.dump(ep_out, f, indent=2, ensure_ascii=False)
        counts[domain] += 1

    print(f"  {condition}_pl: {counts['pl']} eps, {condition}_sw: {counts['sw']} eps "
          f"(under {output_root})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, required=True, help="Path to meme_<condition>.json")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output root; creates <condition>_pl/ and <condition>_sw/ inside")
    args = ap.parse_args()
    unpack(args.input, args.output)


if __name__ == "__main__":
    main()
