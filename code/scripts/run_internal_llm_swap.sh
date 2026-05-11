#!/usr/bin/env bash
# Reproduce tab:llm-ablation (b) — internal-LLM swap on Graphiti, Mem0, MD-flat.
# Answering LLM held at Sonnet 4. 20-episode subset (paper config).
#
# The 20-ep subset is built by dataset_tools/unpack_dataset.py with
# --first-n 20; pass DATA_DIR pointing at that.

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data}"
OUT="${OUT:-output/internal_llm_swap}"
WORKERS="${WORKERS:-4}"
ANSWER_MODEL="${ANSWER_MODEL:-claude-sonnet-4-20250514}"
INTERNAL_MODELS="${INTERNAL_MODELS:-gpt-4.1-mini gpt-5 z-ai/glm-5.1 claude-opus-4-7}"
SYSTEMS="${SYSTEMS:-md_file mem0 graphiti}"

for MODEL in $INTERNAL_MODELS; do
  TAG="${MODEL//\//-}"
  for SYS in $SYSTEMS; do
    case "$SYS" in
      mem0) VENV=mem0_env ;;
      graphiti) VENV=graphiti_env ;;
      *) VENV=main_env ;;
    esac
    source ".venvs/$VENV/bin/activate"
    for DOMAIN in pl sw; do
      echo ""
      echo "=== internal=$TAG | $SYS | $DOMAIN ==="
      python -m eval.run_agent \
        -d "$DATA_DIR/filler32k_$DOMAIN" \
        -o "$OUT/$TAG/$SYS" \
        --agent-type "$SYS" \
        --model "$ANSWER_MODEL" \
        --internal-model "$MODEL" \
        -w "$WORKERS" \
        --skip-existing
    done
    deactivate || true
  done
done

echo "Internal-LLM swap done."
