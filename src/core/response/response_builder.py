from __future__ import annotations

from typing import Any, Dict, List, Sequence

from core.response.citation_generator import CitationGenerator
from core.types import RetrievalResult


class ResponseBuilder:
    def __init__(self, citation_generator: CitationGenerator | None = None):
        self._citation_generator = citation_generator or CitationGenerator()

    def build(
        self, retrieval_results: Sequence[RetrievalResult], query: str
    ) -> Dict[str, Any]:
        normalized_query = _normalize_query(query)
        normalized_results = _normalize_results(retrieval_results)
        if not normalized_results:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "未找到相关文档，请先运行 ingest.py 摄取数据",
                    }
                ],
                "structuredContent": {
                    "query": normalized_query,
                    "citations": [],
                    "results": [],
                },
            }
        citations = self._citation_generator.generate(normalized_results)
        markdown = _build_markdown_answer(normalized_query, normalized_results, citations)
        return {
            "content": [{"type": "text", "text": markdown}],
            "structuredContent": {
                "query": normalized_query,
                "citations": citations,
                "results": [item.to_dict() for item in normalized_results],
            },
        }


def _normalize_query(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must be non-empty")
    return normalized


def _normalize_results(value: Sequence[RetrievalResult]) -> List[RetrievalResult]:
    output: List[RetrievalResult] = []
    for item in value:
        if not isinstance(item, RetrievalResult):
            raise ValueError("retrieval_results must contain RetrievalResult")
        output.append(item)
    return output


def _build_markdown_answer(
    query: str,
    results: Sequence[RetrievalResult],
    citations: Sequence[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("Query: " + query)
    lines.append("")
    for index, item in enumerate(results, start=1):
        citation_index = "[" + str(index) + "]"
        source = citations[index - 1].get("source", "unknown")
        preview = item.text.strip().replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:177] + "..."
        lines.append(
            str(index)
            + ". "
            + preview
            + " "
            + citation_index
            + " (score="
            + format(float(item.score), ".4f")
            + ", source="
            + str(source)
            + ")"
        )
    return "\n".join(lines)
