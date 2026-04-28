from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import Fusion
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


class HybridSearch:
    def __init__(
        self,
        settings: Any,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[Fusion] = None,
    ):
        self._settings = settings
        self._query_processor = query_processor or QueryProcessor()
        self._dense_retriever = dense_retriever or DenseRetriever(settings=settings)
        self._sparse_retriever = sparse_retriever or SparseRetriever(settings=settings)
        self._fusion = fusion or Fusion(settings=settings)
        self._default_top_k = _extract_top_k(settings)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        resolved_top_k = _resolve_top_k(self._default_top_k, top_k)
        processed = self._query_processor.process(query, filters=filters, trace=trace)
        dense_error: Optional[Exception] = None
        sparse_error: Optional[Exception] = None
        dense_results: List[RetrievalResult] = []
        sparse_results: List[RetrievalResult] = []
        try:
            dense_results = self._dense_retriever.retrieve(
                processed.query,
                top_k=resolved_top_k,
                filters=processed.filters,
                trace=trace,
            )
        except Exception as exc:
            dense_error = exc
        try:
            sparse_results = self._sparse_retriever.retrieve(
                processed.keywords,
                top_k=resolved_top_k,
                trace=trace,
            )
        except Exception as exc:
            sparse_error = exc
        if dense_error is not None and sparse_error is not None:
            raise RuntimeError(
                "hybrid search failed: both routes unavailable; dense="
                + str(dense_error)
                + "; sparse="
                + str(sparse_error)
            )
        fused = self._fusion.fuse(
            dense_results,
            sparse_results,
            top_k=resolved_top_k,
            trace=trace,
        )
        filtered = self._apply_metadata_filters(fused, processed.filters)
        output = filtered[:resolved_top_k]
        if trace is not None:
            trace.record_stage(
                "hybrid_search",
                {
                    "status": "ok",
                    "top_k": resolved_top_k,
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "result_count": len(output),
                    "dense_fallback": dense_error is not None,
                    "sparse_fallback": sparse_error is not None,
                },
            )
        return output

    def _apply_metadata_filters(
        self,
        candidates: List[RetrievalResult],
        filters: Mapping[str, Any],
    ) -> List[RetrievalResult]:
        if not filters:
            return list(candidates)
        output: List[RetrievalResult] = []
        for item in candidates:
            if _matches_filters(item.metadata, filters):
                output.append(item)
        return output


def _matches_filters(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, value in filters.items():
        if metadata.get(key) != value:
            return False
    return True


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
