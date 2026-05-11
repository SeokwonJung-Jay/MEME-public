"""
BaseMemorySystem — Abstract interface for all MEME memory systems.

Architecture:
  [Memory System]  ← each system is an independent box
    - ingest_session(session)  → store internally (native method)
    - retrieve(question)       → return context string (query budget = 1)

  [Evaluation Shell]  ← common, outside the system
    - context = system.retrieve(question)
    - answer = unified_llm(context, question)  ← same LLM + prompt for all

Every memory system must implement:
  - ingest_session(): process a conversation session and update memory (native)
  - retrieve(): given a question, return relevant context from memory (query budget = 1)
  - get_memory_snapshot(): return full memory state (for W-check)
  - reset(): clear all state for a new episode
"""

from abc import ABC, abstractmethod


# Unified answer prompt — used by evaluation shell for ALL systems
UNIFIED_ANSWER_PROMPT = """Answer the user's question based ONLY on the context provided below.
If the information is not in the context, say you don't have that information.
Answer with ONLY the value. Do not explain or add context.

Context:
{context}

Question: {question}"""


class BaseMemorySystem(ABC):

    @abstractmethod
    def ingest_session(self, session: dict) -> dict:
        """Process a conversation session and store relevant info in memory.

        Each system uses its own native method (internal LLM, API calls, etc.).

        Args:
            session: {"conversation": [...], "timestamp": "...", "session_id": "...", ...}

        Returns:
            Ingest metadata dict (tool calls, tokens, etc.). Structure is system-specific.
        """
        pass

    @abstractmethod
    def retrieve(self, question: str) -> str:
        """Retrieve relevant context from memory for a given question.

        Query budget = 1: each system does exactly one native retrieval.
        - MD-flat: reads entire memory.md
        - Hermes: MEMORY.md + USER.md snapshot + 1 session_search
        - Mem0: m.search(question)
        - Graphiti: client.search(question)
        - Karpathy: query.py(question)

        Args:
            question: natural language question string

        Returns:
            Context string to be injected into unified answer prompt.
        """
        pass

    def answer_question(self, question: str, client=None, model=None) -> str:
        """Answer a question using unified LLM + unified prompt.

        This is NOT overridden by subclasses. All systems use the same path:
          1. context = self.retrieve(question)
          2. answer = unified_llm(UNIFIED_ANSWER_PROMPT.format(context, question))

        Args:
            question: natural language question string
            client: LLM client (OpenAI-compatible)
            model: model name for the unified answering LLM

        Returns:
            Answer string.
        """
        # Track retrieve / answer scopes separately
        try:
            from eval.budget_tracker import get_tracker
            _tracker = get_tracker()
        except Exception:
            _tracker = None

        if _tracker is not None:
            _tracker.set_scope("retrieve")
        context = self.retrieve(question)
        self._last_retrieved_context = context

        if _tracker is not None:
            _tracker.set_scope("answer")
        prompt = UNIFIED_ANSWER_PROMPT.format(context=context, question=question)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        if _tracker is not None:
            _tracker.set_scope("ingest")  # restore default for any subsequent ingest

        # Log token usage
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            input_tok = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0) or 0
            output_tok = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0) or 0
            if not hasattr(self, '_answer_token_usage'):
                self._answer_token_usage = {"input_tokens": 0, "output_tokens": 0}
            self._answer_token_usage["input_tokens"] += input_tok
            self._answer_token_usage["output_tokens"] += output_tok

        return response.choices[0].message.content.strip()

    def finalize_ingest(self):
        """Called after all sessions in a phase are ingested, before snapshot/questions.

        Override for systems that need post-ingest processing (e.g., Karpathy compile).
        Default: no-op.
        """
        pass

    @abstractmethod
    def get_memory_snapshot(self) -> dict:
        """Return the full memory state for W-check evaluation.

        Must include a "text" key with the entire memory as plain text.

        Returns:
            {"text": str, ...}
        """
        pass

    def get_retrieved_context(self) -> str:
        """Return the context from the most recent retrieve() call.

        Used for R-check: was the gold fact in the context?

        Returns:
            Plain text of retrieved context.
        """
        return getattr(self, '_last_retrieved_context', '')

    @abstractmethod
    def reset(self):
        """Reset all internal state for a new episode."""
        pass
