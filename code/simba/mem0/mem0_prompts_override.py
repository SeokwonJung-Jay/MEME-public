"""Runtime override for mem0's ADDITIVE_EXTRACTION_PROMPT — NO edits to mem0 package.

mem0 v2 reads ADDITIVE_EXTRACTION_PROMPT inline at extraction time:
    system_prompt = ADDITIVE_EXTRACTION_PROMPT
    if is_agent_scoped: system_prompt += AGENT_CONTEXT_SUFFIX
    response = self.llm.generate_response(messages=[
        {"role": "system", "content": system_prompt}, ...
    ])

We install a patch on each Memory instance's `llm.generate_response` that
intercepts the system message and substitutes the currently-set prompt.
The agent-scope suffix (if present in the original system prompt) is preserved.

Usage:
    install = MemoryPromptOverride(memory)
    install.set_prompt(simba_optimized_prompt_text)
    memory.add(...)   # extraction now uses simba_optimized_prompt_text

One MemoryPromptOverride per Memory instance. SIMBA's per-thread Memory
gives us prompt isolation across worker threads without thread-locals
(each Memory has its own LLM client, patched independently).
"""

from mem0.configs.prompts import AGENT_CONTEXT_SUFFIX, ADDITIVE_EXTRACTION_PROMPT


DEFAULT_PROMPT = ADDITIVE_EXTRACTION_PROMPT


class MemoryPromptOverride:
    """Wraps a Memory instance's LLM so the extraction system prompt is
    swappable per call, without touching the underlying mem0 package."""

    def __init__(self, memory):
        self._memory = memory
        self._llm = memory.llm
        self._current_prompt = DEFAULT_PROMPT
        self._orig_generate = self._llm.generate_response
        self._llm.generate_response = self._patched_generate

    def set_prompt(self, prompt: str):
        """Set the system prompt to use for the next extraction calls."""
        self._current_prompt = prompt

    def _patched_generate(self, messages, **kw):
        new_messages = []
        for m in messages:
            if m.get("role") == "system":
                # Preserve agent-context suffix if present in the original.
                content = self._current_prompt
                if AGENT_CONTEXT_SUFFIX.strip() in m.get("content", ""):
                    content += AGENT_CONTEXT_SUFFIX
                new_messages.append({**m, "content": content})
            else:
                new_messages.append(m)
        return self._orig_generate(messages=new_messages, **kw)

    def restore(self):
        """Restore the original llm.generate_response. Useful for cleanup."""
        self._llm.generate_response = self._orig_generate
