from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


class Fusion:
    def __init__(self, settings: Any, rrf_k: Optional[int] = None):
        self._settings = settings
        self._rrf_k = _extract_rrf_k(settings, rrf_k)
        self._default_top_k = _extract_top_k(settings)

    def fuse(
        self,
        dense_results: Sequence[RetrievalResult],
        sparse_results: Sequence[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        normalized_dense = _normalize_results(dense_results, "dense_results")
        normalized_sparse = _normalize_results(sparse_results, "sparse_results")
        resolved_top_k = _resolve_top_k(self._default_top_k, top_k)
        if not normalized_dense and not normalized_sparse:
            if trace is not None:
                trace.record_stage(
                    "fusion",
                    {
                        "status": "ok",
                        "rrf_k": self._rrf_k,
                        "result_count": 0,
                        "top_k": resolved_top_k,
                    },
                )
            return []
        merged: Dict[str, Dict[str, Any]] = {}
        _merge_route(merged, normalized_dense, self._rrf_k)
        _merge_route(merged, normalized_sparse, self._rrf_k)
        ranked = sorted(
            merged.values(),
            key=lambda item: (-float(item["score"]), str(item["chunk_id"])),
        )
        output: List[RetrievalResult] = []
        for item in ranked[:resolved_top_k]:
            output.append(
                RetrievalResult(
                    chunk_id=str(item["chunk_id"]),
                    score=float(item["score"]),
                    text=str(item["text"]),
                    metadata=dict(item["metadata"]),
                )
            )
        if trace is not None:
            trace.record_stage(
                "fusion",
                {
                    "status": "ok",
                    "rrf_k": self._rrf_k,
                    "dense_count": len(normalized_dense),
                    "sparse_count": len(normalized_sparse),
                    "result_count": len(output),
                    "top_k": resolved_top_k,
                },
            )
        return output


def _merge_route(
    merged: Dict[str, Dict[str, Any]],
    route_results: Sequence[RetrievalResult],
    rrf_k: int,
) -> None:
    for rank, result in enumerate(route_results, start=1):
        chunk_id = result.chunk_id
        score_boost = 1.0 / float(rrf_k + rank)
        item = merged.get(chunk_id)
        if item is None:
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "score": score_boost,
                "text": result.text,
                "metadata": dict(result.metadata),
            }
            continue
        item["score"] = float(item["score"]) + score_boost
        if not str(item.get("text", "")) and result.text:
            item["text"] = result.text
        if not isinstance(item.get("metadata"), Mapping) or not item["metadata"]:
            item["metadata"] = dict(result.metadata)


def _normalize_results(value: Any, field_name: str) -> List[RetrievalResult]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(field_name + " must be a sequence of RetrievalResult")
    output: List[RetrievalResult] = []
    for item in value:
        if not isinstance(item, RetrievalResult):
            raise ValueError(field_name + " must contain RetrievalResult")
        output.append(item)
    return output


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


def _extract_rrf_k(settings: Any, override: Optional[int]) -> int:
    if override is not None:
        return _validate_rrf_k(override)
    if isinstance(settings, Mapping):
        retrieval = settings.get("retrieval")
        if isinstance(retrieval, Mapping):
            value = retrieval.get("rrf_k")
            if value is not None:
                return _validate_rrf_k(value)
            fusion = retrieval.get("fusion")
            if isinstance(fusion, Mapping) and "k" in fusion:
                return _validate_rrf_k(fusion.get("k"))
        if "rrf_k" in settings:
            return _validate_rrf_k(settings.get("rrf_k"))
        return 60
    retrieval = getattr(settings, "retrieval", None)
    if retrieval is not None:
        value = getattr(retrieval, "rrf_k", None)
        if value is not None:
            return _validate_rrf_k(value)
        fusion = getattr(retrieval, "fusion", None)
        if fusion is not None:
            fusion_k = getattr(fusion, "k", None)
            if fusion_k is not None:
                return _validate_rrf_k(fusion_k)
    value = getattr(settings, "rrf_k", None)
    if value is not None:
        return _validate_rrf_k(value)
    return 60


def _validate_rrf_k(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("rrf_k must be an integer")
    if value <= 0:
        raise ValueError("rrf_k must be positive")
    return value
