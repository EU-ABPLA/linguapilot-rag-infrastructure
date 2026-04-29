from __future__ import annotations

from typing import Any, Dict, List, Sequence

from core.types import RetrievalResult


class CitationGenerator:
    def generate(self, retrieval_results: Sequence[RetrievalResult]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for item in retrieval_results:
            source = _extract_source(item)
            citation: Dict[str, Any] = {
                "source": source,
                "chunk_id": item.chunk_id,
                "score": float(item.score),
            }
            page = item.metadata.get("page")
            if isinstance(page, int):
                citation["page"] = page
            citations.append(citation)
        return citations


def _extract_source(item: RetrievalResult) -> str:
    for key in ("source_path", "source", "document"):
        value = item.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"
