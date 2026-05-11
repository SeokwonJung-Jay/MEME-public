#!/usr/bin/env bash
# Reproduce tab:sd — 5-seed stability run (paper used MD-flat).
# Each seed reruns the whole main pipeline; outputs go to OUT/seed_<N>/<system>.

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data}"
OUT="${OUT:-output/multi_seed}"
WORKERS="${WORKERS:-4}"
ANSWER_MODEL="${ANSWER_MODEL:-gpt-4.1-mini}"
INTERNAL_MODEL="${INTERNAL_MODEL:-gpt-4.1-mini}"
SYSTEMS="${SYSTEMS:-md_file}"   # space-separated. Default = MD-flat (paper config).
SEEDS="${SEEDS:-1 2 3 4 5}"

source ".venvs/main_env/bin/activate"

for SEED in $SEEDS; do
  for SYS in $SYSTEMS; do
    for DOMAIN in pl sw; do
      echo ""
      echo "=== seed=$SEED | $SYS | $DOMAIN ==="
      python -m eval.run_agent \
        -d "$DATA_DIR/filler32k_$DOMAIN" \
        -o "$OUT/seed_${SEED}/$SYS" \
        --agent-type "$SYS" \
        --model "$ANSWER_MODEL" \
        --internal-model "$INTERNAL_MODEL" \
        --seed "$SEED" \
        -w "$WORKERS" \
        --skip-existing
    done
  done
done

echo ""
echo "All seeds done. Aggregate variance across seed_1..seed_5/<system>/judge."
