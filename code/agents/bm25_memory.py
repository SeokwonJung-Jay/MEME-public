"""BM25Memory — Sparse (lexical) retrieval baseline via bm25s.

Architecture:
  - Ingest: each session → one chunk (session text with timestamp) → append to store
  - Retrieve: question → bm25s top-k chunks → concatenate as context
  - Answer: unified LLM (from base) uses retrieved context

bm25s (https://github.com/jataware/bm25s): fast, sparse-matrix BM25 in pure Python.
No internal LLM for ingest — pure lexical. Cheap and fast baseline.
"""

import logging
from typing import List, Dict

import bm25s

from agents.base import BaseMemorySystem
from agents._chunking import session_to_chunks


class BM25Memory(BaseMemorySystem):
    """BM25 lexical retrieval baseline using bm25s.

    Stores session transcripts as raw chunks. Retrieval = top-k BM25 hits.
    bm25s doesn't support incremental indexing — index is rebuilt per ingest
    (fast: O(N) sparse-matrix construction, ~ms for dozens of docs).
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = None,
                 internal_model: str = None, top_k: int = 5, method: str = "lucene",
                 chunk_max_tokens: int = 4096, **kwargs):
        """top_k: retrieval depth. chunk_max_tokens: split session if longer.
        method: bm25s IDF variant — 'lucene' (default), 'robertson', 'atire', 'bm25l', 'bm25+'."""
        self._chunks: List[str] = []
        self._top_k = top_k
        self._method = method
        self._chunk_max_tokens = chunk_max_tokens
        self._retriever: bm25s.BM25 | None = None
        self._last_retrieved_context = ""

    def _rebuild_index(self):
        """Rebuild the bm25s index from current chunks. bm25s tqdm is too noisy;
        silence via show_progress=False."""
        if not self._chunks:
            self._retriever = None
            return
        self._retriever = bm25s.BM25(method=self._method)
        tokens = bm25s.tokenize(self._chunks, show_progress=False)
        self._retriever.index(tokens, show_progress=False)

    def ingest_session(self, session: Dict) -> Dict:
        new_chunks = session_to_chunks(session, max_tokens=self._chunk_max_tokens)
        self._chunks.extend(new_chunks)
        self._rebuild_index()
        return {"chunks_indexed": len(self._chunks),
                "new_chunks_from_session": len(new_chunks)}

    def retrieve(self, question: str) -> str:
        if self._retriever is None or not self._chunks:
            return "(no memory)"
        q_tokens = bm25s.tokenize([question], show_progress=False)
        k = min(self._top_k, len(self._chunks))
        results, _ = self._retriever.retrieve(q_tokens, k=k, corpus=self._chunks,
                                              show_progress=False)
        # results shape: (n_queries=1, k)
        top_chunks = list(results[0])
        return "\n\n---\n\n".join(top_chunks)

    def get_memory_snapshot(self) -> Dict:
        text = "\n\n---\n\n".join(self._chunks) if self._chunks else "(empty)"
        return {"type": "bm25", "method": self._method, "text": text,
                "n_chunks": len(self._chunks)}

    def reset(self):
        self._chunks = []
        self._retriever = None
        self._last_retrieved_context = ""
