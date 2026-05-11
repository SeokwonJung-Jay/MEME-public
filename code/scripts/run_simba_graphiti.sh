#!/usr/bin/env bash
# Reproduce the SIMBA prompt-optimization ablation on Graphiti.
# Requires a Neo4j cluster (one port per SIMBA worker thread). Start it with
# code/scripts/start_neo4j_cluster.sh before running.
#
# Usage:
#   ./code/scripts/run_simba_graphiti.sh [extra args forwarded to run_simba.py]

set -euo pipefail
cd "$(dirname "$0")/../simba/graphiti"

if [ ! -d ".venv" ]; then
  echo "Creating SIMBA-graphiti venv (Python 3.12+) ..."
  ${PYBIN:-python3.12} -m venv .venv
  source .venv/bin/activate
  pip install -q -U pip
  pip install -q -r requirements.txt
else
  source .venv/bin/activate
fi

for var in OPENAI_API_KEY ANTHROPIC_API_KEY NEO4J_BASE_PORT; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: $var not set (export from code/.env, NEO4J_BASE_PORT must match the cluster you started)"
    exit 1
  fi
done

mkdir -p results
exec python3 -u run_simba.py "$@"
