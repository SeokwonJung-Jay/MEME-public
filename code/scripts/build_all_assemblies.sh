#!/usr/bin/env bash
# Run Stage 5 (assemble_episodes.py) for all 3 variants x 2 domains and lay out
# the directories so dataset_tools/build_dataset.py reads them directly.
#
# Inputs:
#   --conv-dir DIR        Stage 3 conversations root (expects conversations_pl/, conversations_sw/)
#   --filler-pl PATH      PL filler pool JSON (from MEME-fillers/fillers_pl.json)
#   --filler-sw PATH      SW filler pool JSON (from MEME-fillers/fillers_sw.json)
#   --out-root DIR        Output root for the assembled trees
#
# Output layout (consumed by build_dataset.py without manual mv):
#   <out-root>/episodes/{episodes_pl,episodes_sw}/...      (Stage 1, must exist already)
#   <out-root>/nofiller/{nofiller_pl,nofiller_sw}/...
#   <out-root>/filler32k/{filler_pl,filler_sw}/...
#   <out-root>/filler128k/{filler_pl,filler_sw}/...

set -euo pipefail

CONV_DIR=""
FILLER_PL=""
FILLER_SW=""
OUT_ROOT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --conv-dir)   CONV_DIR="$2"; shift 2 ;;
    --filler-pl)  FILLER_PL="$2"; shift 2 ;;
    --filler-sw)  FILLER_SW="$2"; shift 2 ;;
    --out-root)   OUT_ROOT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

for var in CONV_DIR FILLER_PL FILLER_SW OUT_ROOT; do
  if [ -z "${!var}" ]; then
    echo "Error: --${var,,} is required" >&2
    exit 1
  fi
done

# Resolve user-supplied paths to absolute (relative to invocation cwd) before changing cwd.
abs() { python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"; }
CONV_DIR=$(abs "$CONV_DIR")
FILLER_PL=$(abs "$FILLER_PL")
FILLER_SW=$(abs "$FILLER_SW")
OUT_ROOT=$(abs "$OUT_ROOT")

mkdir -p "$OUT_ROOT"

cd "$(dirname "$0")/.."

run_variant() {
  local label="$1"        # nofiller | filler32k | filler128k
  local subdir="$2"        # nofiller | filler
  local extra_args="$3"    # e.g., "-n 0" or "-t 7000" or "-t 25000"

  for domain in personal_life software_project; do
    local prefix=$([ "$domain" = "personal_life" ] && echo "pl" || echo "sw")
    local filler_arg=$([ "$prefix" = "pl" ] && echo "$FILLER_PL" || echo "$FILLER_SW")
    local out_dir="$OUT_ROOT/$label/${subdir}_${prefix}"
    echo ""
    echo "=== $label / $domain -> $out_dir ==="
    python3 data/assemble_episodes.py \
      -c "$CONV_DIR/conversations_${prefix}" \
      -f "$filler_arg" \
      -o "$out_dir" \
      --domain "$domain" \
      $extra_args
  done
}

run_variant nofiller     nofiller "-n 0"
run_variant filler32k    filler   "-t 7000"
run_variant filler128k   filler   "-t 25000"

echo ""
echo "All variants assembled under $OUT_ROOT/"
echo "Now run:"
echo "  python3 dataset_tools/build_dataset.py \\"
echo "    --episodes-dir   $OUT_ROOT/episodes \\"
echo "    --nofiller-dir   $OUT_ROOT/nofiller \\"
echo "    --filler32k-dir  $OUT_ROOT/filler32k \\"
echo "    --filler128k-dir $OUT_ROOT/filler128k"
