"""Runtime prompt overrides for graphiti_core — NO edits to the installed package.

Strategy: monkey-patch the `.func` attribute of selected VersionWrapper instances
inside `graphiti_core.prompts.prompt_library`. The wrappers still apply the
DO_NOT_ESCAPE_UNICODE suffix to system messages; we just swap the inner function.

Thread-safe: each thread has its OWN `CURRENT_INSTRUCTIONS` via `threading.local()`.
SIMBA's ThreadPoolExecutor workers can therefore run different candidate prompts
in parallel without stomping on each other. graphiti-core's internal asyncio
tasks run in the caller's thread, so they see the same thread-local snapshot.
"""
import threading
from typing import Any

from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.models import Message
from graphiti_core.prompts.prompt_helpers import to_prompt_json


# ============================================================
# Default instructions (verbatim from graphiti_core/prompts/*.py, instruction-
# bearing portion only — variable data like <CURRENT MESSAGE> goes in user msg)
# ============================================================

DEFAULT_EXTRACT_MESSAGE_INSTRUCTIONS = """You are an AI assistant that extracts entity nodes from conversational messages.
Your primary task is to extract and classify the speaker and other significant entities mentioned in the conversation.

You are given a conversation context and a CURRENT MESSAGE. Your task is to extract **entity nodes** mentioned **explicitly or implicitly** in the CURRENT MESSAGE.
Pronoun references such as he/she/they or this/that/those should be disambiguated to the names of the
reference entities. Only extract distinct entities from the CURRENT MESSAGE. Don't extract pronouns like you, me, he/she/they, we/us as entities.

1. **Speaker Extraction**: Always extract the speaker (the part before the colon `:` in each dialogue line) as the first entity node.
   - If the speaker is mentioned again in the message, treat both mentions as a **single entity**.

2. **Entity Identification**:
   - Extract all significant entities, concepts, or actors that are **explicitly or implicitly** mentioned in the CURRENT MESSAGE.
   - **Exclude** entities mentioned only in the PREVIOUS MESSAGES (they are for context only).

3. **Entity Classification**:
   - Use the descriptions in ENTITY TYPES to classify each extracted entity.
   - Assign the appropriate `entity_type_id` for each one.

4. **Exclusions**:
   - Do NOT extract entities representing relationships or actions.
   - Do NOT extract dates, times, or other temporal information—these will be handled separately.

5. **Formatting**:
   - Be **explicit and unambiguous** in naming entities (e.g., use full names when available)."""


DEFAULT_EDGE_INSTRUCTIONS = """You are an expert fact extractor that extracts fact triples from text.
1. Extracted fact triples should also be extracted with relevant date information.
2. Treat the CURRENT TIME as the time the CURRENT MESSAGE was sent. All temporal information should be extracted relative to this time.

# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
  and can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.

You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.

# EXTRACTION RULES

1. **Entity Name Validation**: `source_entity_name` and `target_entity_name` must use only the `name` values from the ENTITIES list provided above.
   - **CRITICAL**: Using names not in the list will cause the edge to be rejected
2. Each fact must involve two **distinct** entities.
3. Do not emit duplicate or semantically redundant facts.
4. The `fact` should closely paraphrase the original source sentence(s). Do not verbatim quote the original text.
5. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions (e.g., "last week").
6. Do **not** hallucinate or infer temporal bounds from unrelated events.

# RELATION TYPE RULES

- If FACT_TYPES are provided and the relationship matches one of the types (considering the entity type signature), use that fact_type_name as the `relation_type`.
- Otherwise, derive a `relation_type` from the relationship predicate in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH).

# DATETIME RULES

- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set `valid_at` to REFERENCE_TIME.
- If a change/termination is expressed, set `invalid_at` to the relevant timestamp.
- Leave both fields `null` if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00."""


DEFAULT_DEDUPE_NODES_INSTRUCTIONS = """You are a helpful assistant that determines whether or not ENTITIES extracted from a conversation are duplicates of existing entities.

For each of the above ENTITIES, determine if the entity is a duplicate of any of the EXISTING ENTITIES.

Entities should only be considered duplicates if they refer to the *same real-world object or concept*.

Do NOT mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.

For every entity, return an object with the following keys:
{
    "id": integer id from ENTITIES,
    "name": the best full name for the entity (preserve the original name unless a duplicate has a more complete name),
    "duplicate_name": the name of the EXISTING ENTITY that is the best duplicate match, or empty string if there is no duplicate
}

- Only use names that appear in EXISTING ENTITIES.
- Use empty string if there is no duplicate.
- Never fabricate entity names."""


# ============================================================
# Thread-local current instructions (swapped per candidate per worker thread)
# ============================================================

_tl = threading.local()


def _get_instructions() -> dict[str, str]:
    if not hasattr(_tl, "instructions"):
        _tl.instructions = {
            "extract_message": DEFAULT_EXTRACT_MESSAGE_INSTRUCTIONS,
            "edge": DEFAULT_EDGE_INSTRUCTIONS,
            "dedupe_nodes": DEFAULT_DEDUPE_NODES_INSTRUCTIONS,
        }
    return _tl.instructions


def set_current_instructions(extract_message: str, edge: str, dedupe_nodes: str):
    """Set the three SIMBA-optimizable instructions for the current thread."""
    instructions = _get_instructions()
    instructions["extract_message"] = extract_message
    instructions["edge"] = edge
    instructions["dedupe_nodes"] = dedupe_nodes


# ============================================================
# Override prompt-builder functions — each reads from thread-local state
# ============================================================

def _build_extract_message(context: dict[str, Any]) -> list[Message]:
    sys_prompt = _get_instructions()["extract_message"]
    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

{context.get('custom_extraction_instructions', '')}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def _build_edge(context: dict[str, Any]) -> list[Message]:
    edge_types_section = ''
    if context.get('edge_types'):
        edge_types_section = f"""
<FACT_TYPES>
{to_prompt_json(context['edge_types'])}
</FACT_TYPES>
"""
    sys_prompt = _get_instructions()["edge"]
    user_prompt = f"""
<PREVIOUS_MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT_MESSAGE>

<ENTITIES>
{to_prompt_json(context['nodes'])}
</ENTITIES>

<REFERENCE_TIME>
{context['reference_time']}  # ISO 8601 (UTC); used to resolve relative time mentions
</REFERENCE_TIME>
{edge_types_section}
{context.get('custom_extraction_instructions', '')}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def _build_dedupe_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = _get_instructions()["dedupe_nodes"]
    n = len(context['extracted_nodes'])
    user_prompt = f"""
<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

Each of the following ENTITIES were extracted from the CURRENT MESSAGE.
Each entity in ENTITIES is represented as a JSON object with the following structure:
{{
    id: integer id of the entity,
    name: "name of the entity",
    entity_type: ["Entity", "<optional additional label>", ...],
    entity_type_description: "Description of what the entity type represents"
}}

<ENTITIES>
{to_prompt_json(context['extracted_nodes'])}
</ENTITIES>

<EXISTING ENTITIES>
{to_prompt_json(context['existing_nodes'])}
</EXISTING ENTITIES>

Each entry in EXISTING ENTITIES is an object with the following structure:
{{
    name: "name of the candidate entity",
    entity_types: ["Entity", "<optional additional label>", ...],
    ...<additional attributes such as summaries or metadata>
}}

ENTITIES contains {n} entities with IDs 0 through {n - 1}.
Your response MUST include EXACTLY {n} resolutions with IDs 0 through {n - 1}. Do not skip or add IDs.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


# ============================================================
# Install (idempotent)
# ============================================================

_INSTALLED = False
_install_lock = threading.Lock()


def install_overrides():
    """Monkey-patch prompt_library to use our builder functions. Idempotent."""
    global _INSTALLED
    with _install_lock:
        if _INSTALLED:
            return
        prompt_library.extract_nodes.extract_message.func = _build_extract_message
        prompt_library.extract_edges.edge.func = _build_edge
        prompt_library.dedupe_nodes.nodes.func = _build_dedupe_nodes
        _INSTALLED = True
