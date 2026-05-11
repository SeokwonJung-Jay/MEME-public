#!/usr/bin/env bash
# Reproduce the SIMBA prompt-optimization ablation on Karpathy Wiki.
# Karpathy depends on the upstream claude-memory-compiler clone; see the
# project README for setup. SIMBA uses its own isolated venv.
#
# Usage:
#   ./code/scripts/run_simba_karpathy.sh [extra args forwarded to run_simba.py]

set -euo pipefail
cd "$(dirname "$0")/../simba/karpathy"

if [ ! -d ".venv" ]; then
  echo "Creating SIMBA-karpathy venv (Python 3.12+) ..."
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

if [ ! -d "../../.deps/claude-memory-compiler" ]; then
  echo "ERROR: code/.deps/claude-memory-compiler missing."
  echo "       Clone it with:"
  echo "         git clone https://github.com/coleam00/claude-memory-compiler code/.deps/claude-memory-compiler"
  exit 1
fi

mkdir -p results
exec python3 -u run_simba.py "$@"
