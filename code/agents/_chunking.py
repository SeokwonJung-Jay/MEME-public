"""Shared chunking helper for BM25/Dense memories.

Splits a session into token-bounded chunks, respecting turn boundaries.
Each chunk carries the session timestamp header so retrieved snippets
remain self-describing.
"""
from typing import Dict, List

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(s: str) -> int:
    return len(_enc.encode(s, disallowed_special=()))


def session_to_chunks(session: Dict, max_tokens: int = 4096) -> List[str]:
    """Greedy pack turns into chunks up to max_tokens. Each chunk begins with
    the session header line. Never splits mid-turn (turn boundary respected)."""
    ts = session.get("timestamp", "unknown")
    header = f"[Session: {ts}]"
    header_tok = _count_tokens(header)

    lines: List[str] = []
    for turn in session.get("conversation", []):
        role = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('content', '')}")

    chunks: List[str] = []
    current: List[str] = [header]
    current_tok = header_tok

    for line in lines:
        line_tok = _count_tokens(line)
        # If this line alone would exceed the budget, force it into its own
        # chunk (will be oversized, but keeps turn atomic).
        if line_tok + header_tok > max_tokens:
            # Flush current if it has content beyond header
            if len(current) > 1:
                chunks.append("\n".join(current))
            chunks.append(header + "\n" + line)
            current = [header]
            current_tok = header_tok
            continue
        if current_tok + line_tok > max_tokens and len(current) > 1:
            chunks.append("\n".join(current))
            current = [header, line]
            current_tok = header_tok + line_tok
        else:
            current.append(line)
            current_tok += line_tok

    if len(current) > 1:
        chunks.append("\n".join(current))

    return chunks if chunks else [header]
