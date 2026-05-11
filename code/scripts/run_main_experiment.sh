#!/usr/bin/env bash
# Reproduce the main MEME results table:
#   filler32k condition × 6 systems × 100 episodes
#   answering LLM = gpt-4.1-mini, internal LLM = gpt-4.1-mini
# Cost: ~$200 per full run.
#
# Expected data layout (run dataset_tools/unpack_dataset.py first):
#   $DATA_DIR/filler32k_pl/episode_NNN.json
#   $DATA_DIR/filler32k_sw/episode_NNN.json

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data}"
OUT="${OUT:-output/main_table}"
WORKERS="${WORKERS:-4}"
ANSWER_MODEL="${ANSWER_MODEL:-gpt-4.1-mini}"
INTERNAL_MODEL="${INTERNAL_MODEL:-gpt-4.1-mini}"

mkdir -p "$OUT"

# Per-system venv selection. Each line: <agent-type> <venv-name>
SYSTEMS=(
  "bm25      main_env"
  "dense     main_env"
  "md_file   main_env"
  "karpathy  main_env"
  "graphiti  graphiti_env"
  "mem0      mem0_env"
)

for entry in "${SYSTEMS[@]}"; do
  read -r SYS VENV <<< "$entry"
  for DOMAIN in pl sw; do
    echo ""
    echo "=========================================="
    echo "Running $SYS on $DOMAIN (venv: $VENV)"
    echo "=========================================="
    source ".venvs/$VENV/bin/activate"
    python -m eval.run_agent \
      -d "$DATA_DIR/filler32k_$DOMAIN" \
      -o "$OUT/$SYS" \
      --agent-type "$SYS" \
      --model "$ANSWER_MODEL" \
      --internal-model "$INTERNAL_MODEL" \
      -w "$WORKERS" \
      --skip-existing
    deactivate || true
  done
done

echo ""
echo "All systems done. Judge with:"
echo "  source .venvs/main_env/bin/activate"
echo "  python -m eval.judge -d $OUT/<system> -o $OUT/<system>/judge"
