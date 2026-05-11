# SIMBA prompt-optimization (Section 4.3)

DSPy SIMBA optimizes the ingest / retrieve prompts of each memory system on a small training split, then re-evaluates on a held-out test split. This directory contains a self-contained SIMBA experiment for each of the four prompt-tunable systems in the paper:

| System | Subdir | Optimized signatures |
|---|---|---|
| MD-flat | `mdflat/` | ingest, retrieve (answer prompt frozen) |
| Karpathy Wiki | `karpathy/` | flush, compile, query |
| Graphiti | `graphiti/` | extract\_nodes, extract\_edges, dedupe\_nodes |
| Mem0 | `mem0/` | additive\_extraction (the only on-path prompt in mem0 v2) |

The optimized prompts produced by each run are reproduced verbatim in the paper appendix.

## Why isolated venvs

Each system pins different versions of DSPy / mem0ai / graphiti-core / claude-memory-compiler that conflict if installed into a single environment. Each subdir therefore has its own `requirements.txt` and creates a `.venv/` on first run. Use Python 3.12 or newer (override with `PYBIN=python3.13` if needed).

## Common prerequisites

- The released dataset unpacked into `code/data/filler32k_pl/` and `code/data/filler32k_sw/` (see the project README for `dataset_tools/unpack_dataset.py`).
- `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` exported (or sourced from `code/.env`).
- `code/eval/judge.py` available — used as the U-check metric inside SIMBA.

System-specific extras:

- **Karpathy**: requires the upstream `claude-memory-compiler` cloned at `code/.deps/claude-memory-compiler` (same prerequisite as the regular Karpathy run).
- **Graphiti**: requires a Neo4j cluster with one port per SIMBA worker thread. Start it with `code/scripts/start_neo4j_cluster.sh` and export `NEO4J_BASE_PORT` before launching.
- **Mem0**: requires Qdrant (`docker compose -f code/docker-compose.yml up -d`).

## Running

Use the wrapper scripts under `code/scripts/`. They auto-create the per-system venv on first invocation and forward any extra flags to `run_simba.py`:

```bash
./code/scripts/run_simba_mdflat.sh
./code/scripts/run_simba_karpathy.sh
./code/scripts/run_simba_graphiti.sh
./code/scripts/run_simba_mem0.sh
```

Defaults match the paper's single-seed config: `--train 10 --test 10 --seed 7`. The `tab:simba-stability` MD-flat multi-seed numbers come from re-running with `--seed` $\in$ \{1,2,3,4,5\}.

## Cost note

Each optimization run includes a baseline eval, the SIMBA compile loop (which calls the prompt-generator LM and re-evaluates candidates), and a final test eval. With the paper config (10 train / 30 test episodes, gpt-4.1-mini as task LM, Sonnet 4 as answer LM), one full run takes roughly 30–90 minutes and \$30–\$80 in API spend depending on the system.

## Output layout

Each run writes a timestamped folder under `simba/<system>/results/run_YYYYMMDD_HHMMSS/`:

- `optimized_prompts.json` — the final selected ingest/retrieve prompts
- `all_candidates.json` — every candidate SIMBA evaluated, with per-iteration scores
- `report.json` — baseline vs optimized scores on train/test, per-episode breakdown
- `tool_calls.jsonl` — every tool call made during the run (MD-flat / Karpathy)
- `run.log` — full SIMBA log including prompt-model output (see `simba_patches.py`)
