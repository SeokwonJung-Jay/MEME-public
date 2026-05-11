"""DenseMemory — Dense (semantic) retrieval baseline using text-embedding-3-small.

Architecture:
  - Ingest: each session → embed (OpenAI text-embedding-3-small) → append vector
  - Retrieve: question → embed → cosine similarity vs all chunks → top-k
  - Answer: unified LLM (from base) uses retrieved context

No internal LLM for ingest — pure embedding. Cheap: text-embedding-3-small at $0.02/1M.
"""

import os
from typing import List, Dict

import numpy as np

from agents.base import BaseMemorySystem
from agents._chunking import session_to_chunks


class DenseMemory(BaseMemorySystem):
    """Dense (embedding) retrieval baseline using OpenAI text-embedding-3-small (1536-d).

    Stores session transcripts as raw chunks + dense vectors. Retrieval =
    top-k cosine similarity.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = None,
                 internal_model: str = None, embedding_model: str = "text-embedding-3-small",
                 top_k: int = 5, chunk_max_tokens: int = 4096, **kwargs):
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._chunk_max_tokens = chunk_max_tokens
        self._chunks: List[str] = []
        self._embeddings: np.ndarray | None = None  # shape (n, dim)
        self._last_retrieved_context = ""
        self._client = None

    def _openai(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    _EMBED_MAX_TOKENS = 8000  # text-embedding-3-small hard limit 8192, leave margin
    _EMBED_OVERLAP = 400      # 5% overlap — standard RAG default

    @classmethod
    def _split_for_embedding(cls, text: str) -> List[str]:
        """Sliding-window split for texts exceeding the embedding input limit.
        Single chunk → returned as [text]. Oversized → multiple overlapping windows.
        """
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text, disallowed_special=())
        if len(toks) <= cls._EMBED_MAX_TOKENS:
            return [text]
        windows = []
        start = 0
        step = cls._EMBED_MAX_TOKENS - cls._EMBED_OVERLAP
        while start < len(toks):
            end = min(start + cls._EMBED_MAX_TOKENS, len(toks))
            windows.append(enc.decode(toks[start:end]))
            if end >= len(toks):
                break
            start += step
        return windows

    def _embed(self, texts: List[str]) -> np.ndarray:
        client = self._openai()
        resp = client.embeddings.create(model=self._embedding_model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # L2-normalize for cosine-via-dot
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms

    def ingest_session(self, session: Dict) -> Dict:
        raw_chunks = session_to_chunks(session, max_tokens=self._chunk_max_tokens)
        # Split any chunk exceeding the embedding API's 8192-tok input limit
        # into overlapping windows. Usually a no-op; only triggers on outlier
        # sessions (e.g., SW filler with a single >8K code-block turn).
        new_chunks = [w for c in raw_chunks for w in self._split_for_embedding(c)]
        if not new_chunks:
            return {"chunks_indexed": len(self._chunks), "new_chunks_from_session": 0}
        vecs = self._embed(new_chunks)  # (n_new, dim)
        self._chunks.extend(new_chunks)
        if self._embeddings is None:
            self._embeddings = vecs
        else:
            self._embeddings = np.vstack([self._embeddings, vecs])
        return {"chunks_indexed": len(self._chunks),
                "new_chunks_from_session": len(new_chunks)}

    def retrieve(self, question: str) -> str:
        if self._embeddings is None or not self._chunks:
            return "(no memory)"
        q_vec = self._embed([question])  # (1, dim), normalized
        # numpy 2.0 + macOS Accelerate emits spurious divide/overflow warnings during
        # matmul; scores are still valid cosine similarities.
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            scores = (self._embeddings @ q_vec.T).flatten()
        k = min(self._top_k, len(self._chunks))
        top_idx = np.argsort(-scores)[:k]
        top_chunks = [self._chunks[i] for i in top_idx]
        return "\n\n---\n\n".join(top_chunks)

    def get_memory_snapshot(self) -> Dict:
        text = "\n\n---\n\n".join(self._chunks) if self._chunks else "(empty)"
        return {
            "type": "dense",
            "text": text,
            "n_chunks": len(self._chunks),
            "embedding_model": self._embedding_model,
        }

    def reset(self):
        self._chunks = []
        self._embeddings = None
        self._last_retrieved_context = ""
