#!/usr/bin/env bash
# Reproduce tab:main_results "In-context (no memory)" rows.
# Bypasses every memory system; feeds the full episode transcript directly
# to the answering LLM. Distinct from golden_memory.py (gold-fact ceiling).
#
# Usage:
#   ./code/scripts/run_in_context_baseline.sh                       # default: gpt-4.1-mini
#   ANSWER_MODEL=claude-sonnet-4-6 ./code/scripts/run_in_context_baseline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data}"
OUT="${OUT:-output/in_context}"
WORKERS="${WORKERS:-4}"
ANSWER_MODEL="${ANSWER_MODEL:-gpt-4.1-mini}"

source ".venvs/main_env/bin/activate"

TAG="${ANSWER_MODEL//\//-}"
for DOMAIN in pl sw; do
  echo ""
  echo "=== in-context baseline | $ANSWER_MODEL | $DOMAIN ==="
  python -m eval.in_context_baseline \
    -d "$DATA_DIR/filler32k_$DOMAIN" \
    -o "$OUT/$TAG" \
    --model "$ANSWER_MODEL" \
    -w "$WORKERS" \
    --skip-existing
done

echo ""
echo "Done. Judge with:"
echo "  python -m eval.judge -d $OUT/$TAG -o $OUT/$TAG/judge"
