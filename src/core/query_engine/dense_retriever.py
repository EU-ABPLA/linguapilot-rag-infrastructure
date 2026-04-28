from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class DenseRetriever:
    def __init__(
        self,
        settings: Any,
        embedding_client: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        self._settings = settings
        self._embedding_client = embedding_client or EmbeddingFactory.create(settings)
        self._vector_store = vector_store or VectorStoreFactory.create(settings)
        self._default_top_k = _extract_top_k(settings)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        normalized_query = _normalize_query(query)
        resolved_top_k = _resolve_top_k(self._default_top_k, top_k)
        if filters is not None and not isinstance(filters, Mapping):
            raise ValueError("filters must be a mapping")
        vectors = self._embedding_client.embed([normalized_query], trace=trace)
        if len(vectors) != 1:
            raise RuntimeError(
                "dense retriever error: embedding result mismatch, expected 1 got "
                + str(len(vectors))
            )
        vector = vectors[0]
        if not isinstance(vector, Sequence) or len(vector) == 0:
            raise RuntimeError("dense retriever error: empty query embedding")
        records = self._vector_store.query(
            vector,
            top_k=resolved_top_k,
            filters=filters,
            trace=trace,
        )
        results: List[RetrievalResult] = []
        for item in records:
            if not isinstance(item, Mapping):
                continue
            chunk_id = _extract_chunk_id(item)
            if chunk_id is None:
                continue
            text = _extract_text(item)
            score = _extract_score(item)
            metadata = _extract_metadata(item)
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=text,
                    metadata=metadata,
                )
            )
        if trace is not None:
            trace.record_stage(
                "dense_retriever",
                {
                    "status": "ok",
                    "top_k": resolved_top_k,
                    "result_count": len(results),
                },
            )
        return results


def _normalize_query(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must be non-empty")
    return normalized


def _resolve_top_k(default_top_k: int, top_k: Optional[int]) -> int:
    if top_k is None:
        return default_top_k
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return top_k


def _extract_top_k(settings: Any) -> int:
    if isinstance(settings, Mapping):
        retrieval = settings.get("retrieval")
        if isinstance(retrieval, Mapping):
            value = retrieval.get("top_k")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        value = settings.get("top_k")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return 5
    retrieval = getattr(settings, "retrieval", None)
    if retrieval is None:
        return 5
    value = getattr(retrieval, "top_k", 5)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 5


def _extract_chunk_id(item: Mapping[str, Any]) -> Optional[str]:
    raw = item.get("chunk_id", item.get("id"))
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _extract_text(item: Mapping[str, Any]) -> str:
    value = item.get("text")
    if isinstance(value, str):
        return value
    alt = item.get("content")
    if isinstance(alt, str):
        return alt
    return ""


def _extract_score(item: Mapping[str, Any]) -> float:
    value = item.get("score", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _extract_metadata(item: Mapping[str, Any]) -> Dict[str, Any]:
    value = item.get("metadata", {})
    if isinstance(value, Mapping):
        return dict(value)
    return {}
