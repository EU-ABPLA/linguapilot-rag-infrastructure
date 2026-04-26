from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from core.trace.trace_context import TraceContext
from core.types import Chunk

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class SparseEncoder:
    def encode(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            counts = Counter(tokens)
            output.append(
                {
                    "chunk_id": chunk.id,
                    "doc_length": len(tokens),
                    "term_weights": dict(counts),
                }
            )
        if trace is not None:
            trace.record_stage(
                "sparse_encoder",
                {
                    "status": "ok",
                    "chunk_count": len(chunks),
                    "non_empty_docs": sum(1 for item in output if item["doc_length"] > 0),
                },
            )
        return output


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]
