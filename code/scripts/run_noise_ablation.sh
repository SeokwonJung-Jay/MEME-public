#!/usr/bin/env bash
# Reproduce fig:noise-detail — three filler conditions (no filler, 32K, 128K)
# on the highest-overall system per paradigm: MD-flat, Mem0, dense.
# Datasets: dataset_tools/unpack_dataset.py must already have produced
#   data/nofiller_{pl,sw}, data/filler32k_{pl,sw}, data/filler128k_{pl,sw}.

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data}"
OUT="${OUT:-output/noise_ablation}"
WORKERS="${WORKERS:-4}"
ANSWER_MODEL="${ANSWER_MODEL:-gpt-4.1-mini}"
INTERNAL_MODEL="${INTERNAL_MODEL:-gpt-4.1-mini}"
CONDITIONS="${CONDITIONS:-nofiller filler32k filler128k}"
SYSTEMS="${SYSTEMS:-md_file mem0 dense}"

for COND in $CONDITIONS; do
  for SYS in $SYSTEMS; do
    case "$SYS" in
      mem0) VENV=mem0_env ;;
      graphiti) VENV=graphiti_env ;;
      *) VENV=main_env ;;
    esac
    source ".venvs/$VENV/bin/activate"
    for DOMAIN in pl sw; do
      echo ""
      echo "=== $COND | $SYS | $DOMAIN ==="
      python -m eval.run_agent \
        -d "$DATA_DIR/${COND}_$DOMAIN" \
        -o "$OUT/$COND/$SYS" \
        --agent-type "$SYS" \
        --model "$ANSWER_MODEL" \
        --internal-model "$INTERNAL_MODEL" \
        -w "$WORKERS" \
        --skip-existing
    done
    deactivate || true
  done
done

echo "Noise ablation done."
