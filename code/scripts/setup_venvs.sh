#!/usr/bin/env bash
# Set up three isolated Python virtual environments:
#   .venvs/main_env       - BM25, dense, MD-flat, Karpathy, judge, oracle
#   .venvs/graphiti_env   - Graphiti (neo4j-backed)
#   .venvs/mem0_env       - Mem0 (qdrant-backed)
#
# Each env gets its own dependency set to avoid version conflicts.
# Re-run safely; existing venvs are reused.

set -euo pipefail
cd "$(dirname "$0")/.."

PYBIN="${PYBIN:-python3}"

# Python 3.12+ required: claude-memory-compiler pins >=3.12; graphiti-core, mem0ai pin >=3.10.
if ! "$PYBIN" -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
  echo "Error: Python 3.12+ required, got: $("$PYBIN" --version 2>&1 || echo "$PYBIN not found")" >&2
  echo "Set PYBIN to a 3.12+ interpreter, e.g.: PYBIN=python3.12 ./scripts/setup_venvs.sh" >&2
  exit 1
fi

mk_venv() {
  local name="$1"
  local extra_req="$2"
  if [ ! -d ".venvs/$name" ]; then
    echo "Creating .venvs/$name"
    "$PYBIN" -m venv ".venvs/$name"
  fi
  ".venvs/$name/bin/pip" install --upgrade pip wheel
  ".venvs/$name/bin/pip" install -r requirements.txt
  if [ -n "$extra_req" ]; then
    ".venvs/$name/bin/pip" install -r "$extra_req"
  fi
  echo "  $name ready: source .venvs/$name/bin/activate"
}

mk_venv main_env     ""
mk_venv graphiti_env requirements/graphiti.txt
mk_venv mem0_env     requirements/mem0.txt

echo ""
echo "All venvs ready. Activate with:"
echo "  source .venvs/main_env/bin/activate"
echo "  source .venvs/graphiti_env/bin/activate"
echo "  source .venvs/mem0_env/bin/activate"
