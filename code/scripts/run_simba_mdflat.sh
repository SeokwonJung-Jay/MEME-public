#!/usr/bin/env bash
# Reproduce the SIMBA prompt-optimization ablation on MD-flat.
# Each optimization run incurs substantial API cost; see code/simba/README.md
# for paper-config defaults and per-system caveats.
#
# Usage:
#   ./code/scripts/run_simba_mdflat.sh [extra args forwarded to run_simba.py]
#   ./code/scripts/run_simba_mdflat.sh --train 10 --test 30 --seed 42 --max-steps 3

set -euo pipefail
cd "$(dirname "$0")/../simba/mdflat"

if [ ! -d ".venv" ]; then
  echo "Creating SIMBA-mdflat venv (Python 3.12+) ..."
  ${PYBIN:-python3.12} -m venv .venv
  source .venv/bin/activate
  pip install -q -U pip
  pip install -q -r requirements.txt
else
  source .venv/bin/activate
fi

for var in OPENAI_API_KEY ANTHROPIC_API_KEY; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: $var not set (export it from code/.env or your shell)"
    exit 1
  fi
done

mkdir -p results
exec python3 -u run_simba.py "$@"
