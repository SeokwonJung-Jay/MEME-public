---
license: cc-by-4.0
task_categories:
- question-answering
language:
- en
tags:
- llm-agents
- memory
- benchmark
- long-term-memory
- dependency-reasoning
size_categories:
- n<1K
configs:
- config_name: filler32k
  data_files: meme_filler32k.json
  default: true
- config_name: filler128k
  data_files: meme_filler128k.json
- config_name: nofiller
  data_files: meme_nofiller.json
---

# MEME: Multi-Entity and Evolving Memory Evaluation

A benchmark for evaluating LLM memory systems along two orthogonal dimensions: **entity scope** (single vs. multi-entity) and **temporal dynamics** (static vs. evolving). MEME defines six tasks targeting memory-intensive operations in each quadrant, including two task types that no prior benchmark covers: **Cascade** (propagating updates through dependency rules) and **Absence** (recognizing uncertainty when a previously valid answer becomes untrustworthy).

## Dataset summary

- 100 evaluation episodes (50 Personal Life + 50 Software Project)
- Each episode is a chronological sequence of conversational sessions with associated test questions
- Six task types: Exact Recall, Aggregation, Tracking, Deletion, Cascade, Absence
- Conditional dependency rules between entities (e.g., "if health condition changes, switch medication to Thrynexol") let Cascade and Absence questions test logical consistency over time
- All entity values are fictitious to prevent parametric-knowledge contamination

## Configurations (variants)

| Config       | Episodes | Filler tokens | Use case                          |
|--------------|----------|---------------|-----------------------------------|
| `filler32k`  | 100      | ~32K          | Default benchmark setting         |
| `filler128k` | 40       | ~128K         | Stress test under heavy noise (subset of filler32k for tractable cost) |
| `nofiller`   | 100      | none          | Evidence-only sessions            |

## Loading

The recommended way is to download the JSON file directly with `huggingface_hub`. The episode schema includes nested heterogeneous types (e.g., `entity_values` mixes lists and strings depending on task), which the standard `datasets.load_dataset` Arrow path does not handle cleanly.

```python
import json
from huggingface_hub import hf_hub_download

# Default (filler32k)
path = hf_hub_download("meme-benchmark/MEME", "meme_filler32k.json", repo_type="dataset")
episodes = json.load(open(path))

print(f"Loaded {len(episodes)} episodes")
ep = episodes[0]
print(f"First episode: {ep['episode_id']}, tasks: {len(ep['tasks'])}")
```

Other variants:

```python
hf_hub_download("meme-benchmark/MEME", "meme_filler128k.json", repo_type="dataset")
hf_hub_download("meme-benchmark/MEME", "meme_nofiller.json",   repo_type="dataset")
```

## Episode schema

All three variants share the same schema. Each episode is a JSON object with:

- `episode_id` — `pl_NNN` or `sw_NNN`
- `domain` — `personal_life` or `software_project`
- `root` — root entity for the cascade chain
- `root_change` — value transition triggering cascade resolution
- `chain_entities` / `filler_entities` / `entities` — entities used in the episode
- `has_2hop` — whether the cascade chain reaches 2-hop dependents
- `dependency_edges_used` — edges activated for this episode
- `tasks` — list of `{type, target_entities, entity_values, question_template, gold_answer, notes}`. Cas/Abs entries also include `hop` (1 or 2).
- `total_sessions`, `evidence_sessions`, `filler_sessions` — session counts
- `total_tokens`, `evidence_tokens`, `filler_tokens` — token counts
- `evidence_session_indices` — positions of evidence sessions inside `sessions`
- `sessions` — chronological list of conversational sessions
- `before_questions` / `after_questions` — questions asked before/after the upstream change event (used for trivial-pass filtering)

## Trivial-pass filtering

Cascade, Absence, and Deletion task scoring uses a trivial-pass filter: a response counts as correct only if the system also answered the corresponding `before_questions` (pre-change state-check) correctly. This rules out false positives from systems that never encoded the original fact.

## Task types

Tasks in the dataset use abbreviated tags. Cascade and Absence tasks additionally carry a `hop` field (1 or 2) indicating the dependency-chain depth.

| Tag    | Full name      | Quadrant                | What it tests |
|--------|----------------|-------------------------|---------------|
| `ER`   | Exact Recall   | Single-entity, Static   | Verbatim reproduction of a static fact |
| `Agg`  | Aggregation    | Multi-entity, Static    | Combining facts scattered across sessions |
| `Tr`   | Tracking       | Single-entity, Evolving | Reconstructing the revision history of a single entity |
| `Del`  | Deletion       | Single-entity, Evolving | Stopping reporting a fact after explicit user removal |
| `Cas`  | Cascade        | Multi-entity, Evolving  | Propagating updates through a stated dependency rule (`hop` $\in$ \{1, 2\}) |
| `Abs`  | Absence        | Multi-entity, Evolving  | Recognizing uncertainty when no replacement rule applies (`hop` $\in$ \{1, 2\}) |

## Construction

Episodes are generated from hand-crafted DAG knowledge graphs (one per domain) using a five-step pipeline:

1. **Entity set selection** — root + descendants + outside sample
2. **Value assignment** — initial values from per-entity pools, with consistency post-pass
3. **Task assignment** — entities mapped to task types based on topological role
4. **Verbalization** — facts converted to multi-turn dialogues via LLM self-chat
5. **Haystack assembly** — evidence sessions interleaved with filler sessions

Verbalization uses gpt-4o self-chat between a User LLM and an Assistant LLM. Filler conflict filtering combines BM25 + `text-embedding-3-small` hybrid retrieval (top-K=10 candidate surfacing) with a gpt-4o-mini LLM judge. Dataset verification uses a two-layer pipeline (gpt-4o annotation + Gemini 2.5 Flash semantic audit). Full prompts and the construction script are released alongside this dataset.

The filtered filler pools used in haystack assembly are released separately at [`meme-benchmark/MEME-fillers`](https://huggingface.co/datasets/meme-benchmark/MEME-fillers) (1,009 PL sessions from LongMemEval, 9,008 SW sessions from ShareGPT 52K).

## Citation

```bibtex
@misc{meme2026,
  title  = {{MEME}: Multi-Entity and Evolving Memory Evaluation},
  author = {Jung, Seokwon and others},
  year   = {2026}
}
```

## License

Released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. You may share and adapt the dataset for any purpose with appropriate attribution.
