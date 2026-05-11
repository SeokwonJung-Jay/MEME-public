#!/usr/bin/env bash
# Reproduce tab:topk-sweep — top-k ∈ {5, 10, 20, 40} on bm25, dense, and mem0.
# Answering LLM = Sonnet 4 (paper config). Runs on filler32k (40-episode subset
# is implicit in the eval/judge step that filters to knew_but_failed cases).

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data}"
OUT="${OUT:-output/topk_sweep}"
WORKERS="${WORKERS:-4}"
ANSWER_MODEL="${ANSWER_MODEL:-claude-sonnet-4-20250514}"

# Per-system venv (mem0 has its own pinned environment).
declare -A VENV_FOR
VENV_FOR[bm25]=main_env
VENV_FOR[dense]=main_env
VENV_FOR[mem0]=mem0_env

for K in 5 10 20 40; do
  for SYS in bm25 dense mem0; do
    source ".venvs/${VENV_FOR[$SYS]}/bin/activate"
    for DOMAIN in pl sw; do
      echo ""
      echo "=== top_k=$K | $SYS | $DOMAIN ==="
      python -m eval.run_agent \
        -d "$DATA_DIR/filler32k_$DOMAIN" \
        -o "$OUT/${SYS}_top${K}" \
        --agent-type "$SYS" \
        --model "$ANSWER_MODEL" \
        --top-k "$K" \
        -w "$WORKERS" \
        --skip-existing
    done
    deactivate || true
  done
done

echo ""
echo "All top-k cells done. Judge each cell:"
echo "  for k in 5 10 20 40; do for sys in bm25 dense mem0; do"
echo "    python -m eval.judge -d $OUT/\${sys}_top\${k} -o $OUT/\${sys}_top\${k}/judge"
echo "  done; done"
