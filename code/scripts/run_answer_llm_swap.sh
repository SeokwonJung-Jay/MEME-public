#!/usr/bin/env bash
# Reproduce tab:llm-ablation (a) — answer-LLM swap.
# Internal LLM stays gpt-4.1-mini for all systems; only the answer LLM changes.
#
# This re-uses ingested memory snapshots from the main run (output/main_table)
# and only re-runs the answer step. Falls back to a full re-run if main outputs
# are not present.

set -euo pipefail
cd "$(dirname "$0")/.."

MAIN_OUT="${MAIN_OUT:-output/main_table}"
OUT="${OUT:-output/answer_llm_swap}"
WORKERS="${WORKERS:-4}"
ANSWER_MODELS="${ANSWER_MODELS:-claude-sonnet-4-20250514 claude-sonnet-4-6}"

if [[ ! -d "$MAIN_OUT" ]]; then
  echo "Main run outputs not found at $MAIN_OUT — run scripts/run_main_experiment.sh first."
  exit 1
fi

source ".venvs/main_env/bin/activate"

for MODEL in $ANSWER_MODELS; do
  TAG="${MODEL//\//-}"
  for SYS in bm25 dense md_file karpathy mem0 graphiti; do
    echo ""
    echo "=== answer=$TAG | $SYS ==="
    python -m eval.reanswer \
      -d "$MAIN_OUT/$SYS" \
      -o "$OUT/$TAG/$SYS" \
      --model "$MODEL" \
      -w "$WORKERS" \
      --skip-existing
  done
done

echo "Answer-LLM swap done."
