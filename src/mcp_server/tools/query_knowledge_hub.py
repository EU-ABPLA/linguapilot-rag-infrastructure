from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.reranker import Reranker
from core.response.response_builder import ResponseBuilder
from core.settings import load_settings
from core.types import RetrievalResult


def get_tool_schema() -> Dict[str, Any]:
    return {
        "name": "query_knowledge_hub",
        "description": "Search knowledge base and return answer with citations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "collection": {"type": "string"},
            },
            "required": ["query"],
        },
    }


def call_query_knowledge_hub(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    query = arguments.get("query")
    top_k = arguments.get("top_k")
    collection = arguments.get("collection")
    return query_knowledge_hub(query=query, top_k=top_k, collection=collection)


def query_knowledge_hub(
    query: Any,
    top_k: Any = None,
    collection: Any = None,
    settings: Any = None,
    hybrid_search: Optional[HybridSearch] = None,
    reranker: Optional[Reranker] = None,
    response_builder: Optional[ResponseBuilder] = None,
) -> Dict[str, Any]:
    normalized_query = _normalize_query(query)
    resolved_top_k = _resolve_top_k(top_k)
    filters: Dict[str, Any] = {}
    if isinstance(collection, str) and collection.strip():
        filters["collection"] = collection.strip()
    active_settings = settings or load_settings("config/settings.yaml")
    active_hybrid = hybrid_search or HybridSearch(active_settings)
    active_reranker = reranker or Reranker(active_settings)
    builder = response_builder or ResponseBuilder()
    candidates = active_hybrid.search(
        normalized_query,
        top_k=resolved_top_k,
        filters=filters,
    )
    reranked = active_reranker.rerank(
        normalized_query,
        candidates,
        top_k=resolved_top_k,
    )
    return builder.build(reranked, normalized_query)


def _normalize_query(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must be non-empty")
    return normalized


def _resolve_top_k(top_k: Any) -> int:
    if top_k is None:
        return 5
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return top_k
