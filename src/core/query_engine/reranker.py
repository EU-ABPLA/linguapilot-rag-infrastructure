from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker
from libs.reranker.reranker_factory import RerankerFactory


class Reranker:
    def __init__(
        self,
        settings: Any,
        backend: Optional[BaseReranker] = None,
    ):
        self._settings = settings
        self._backend = backend or RerankerFactory.create(settings)
        self._enabled = _extract_enabled(settings)
        self._default_top_k = _extract_top_k(settings)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        normalized_query = _normalize_query(query)
        normalized_candidates = _normalize_candidates(candidates)
        resolved_top_k = _resolve_top_k(self._default_top_k, top_k)
        if not normalized_candidates:
            return []
        if not self._enabled:
            output = normalized_candidates[:resolved_top_k]
            if trace is not None:
                _record_rerank_stage(
                    trace=trace,
                    details={
                        "status": "skipped",
                        "reason": "disabled",
                        "fallback": False,
                        "method": "cross_encoder",
                        "provider": _extract_provider(self._settings),
                        "result_count": len(output),
                    },
                )
            return output
        payload = _to_payload(normalized_candidates)
        try:
            ranked = self._backend.rerank(normalized_query, payload, trace=trace)
            output = _from_payload(ranked, normalized_candidates, resolved_top_k)
            if trace is not None:
                _record_rerank_stage(
                    trace=trace,
                    details={
                        "status": "ok",
                        "fallback": False,
                        "method": "cross_encoder",
                        "provider": _extract_provider(self._settings),
                        "input_count": len(normalized_candidates),
                        "result_count": len(output),
                    },
                )
            return output
        except Exception as exc:
            output = _mark_fallback(normalized_candidates[:resolved_top_k], str(exc))
            if trace is not None:
                _record_rerank_stage(
                    trace=trace,
                    details={
                        "status": "fallback",
                        "fallback": True,
                        "reason": str(exc),
                        "method": "cross_encoder",
                        "provider": _extract_provider(self._settings),
                        "input_count": len(normalized_candidates),
                        "result_count": len(output),
                    },
                )
            return output


def _normalize_query(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must be non-empty")
    return normalized


def _normalize_candidates(candidates: Any) -> List[RetrievalResult]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError("candidates must be a sequence of RetrievalResult")
    output: List[RetrievalResult] = []
    for item in candidates:
        if not isinstance(item, RetrievalResult):
            raise ValueError("candidates must contain RetrievalResult")
        output.append(item)
    return output


def _to_payload(candidates: Sequence[RetrievalResult]) -> List[Mapping[str, Any]]:
    payload: List[Mapping[str, Any]] = []
    for item in candidates:
        payload.append(
            {
                "id": item.chunk_id,
                "content": item.text,
                "score": float(item.score),
                "metadata": dict(item.metadata),
            }
        )
    return payload


def _from_payload(
    ranked: Sequence[Mapping[str, Any]],
    original: Sequence[RetrievalResult],
    top_k: int,
) -> List[RetrievalResult]:
    by_id: Dict[str, RetrievalResult] = {item.chunk_id: item for item in original}
    output: List[RetrievalResult] = []
    used = set()
    for item in ranked:
        if not isinstance(item, Mapping):
            continue
        raw_id = item.get("id", item.get("chunk_id"))
        if not isinstance(raw_id, str):
            continue
        chunk_id = raw_id.strip()
        if not chunk_id or chunk_id in used:
            continue
        source = by_id.get(chunk_id)
        if source is None:
            continue
        output.append(source)
        used.add(chunk_id)
    for item in original:
        if item.chunk_id in used:
            continue
        output.append(item)
        used.add(item.chunk_id)
    return output[:top_k]


def _mark_fallback(candidates: Sequence[RetrievalResult], reason: str) -> List[RetrievalResult]:
    output: List[RetrievalResult] = []
    for item in candidates:
        metadata = dict(item.metadata)
        metadata["rerank_fallback"] = True
        metadata["rerank_fallback_reason"] = reason
        output.append(
            RetrievalResult(
                chunk_id=item.chunk_id,
                score=float(item.score),
                text=item.text,
                metadata=metadata,
            )
        )
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


def _extract_enabled(settings: Any) -> bool:
    if isinstance(settings, Mapping):
        rerank = settings.get("rerank")
        if isinstance(rerank, Mapping):
            value = rerank.get("enabled")
            if isinstance(value, bool):
                return value
        value = settings.get("enabled")
        if isinstance(value, bool):
            return value
        return True
    rerank = getattr(settings, "rerank", None)
    if rerank is None:
        return True
    value = getattr(rerank, "enabled", True)
    if isinstance(value, bool):
        return value
    return True


def _extract_provider(settings: Any) -> str:
    if isinstance(settings, Mapping):
        rerank = settings.get("rerank")
        if isinstance(rerank, Mapping):
            value = rerank.get("provider")
            if isinstance(value, str) and value.strip():
                return value.strip()
        value = settings.get("provider")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "unknown"
    rerank = getattr(settings, "rerank", None)
    if rerank is not None:
        value = getattr(rerank, "provider", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = getattr(settings, "provider", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown"


def _record_rerank_stage(trace: TraceContext, details: Dict[str, Any]) -> None:
    trace.record_stage("rerank", dict(details))
    trace.record_stage("reranker", dict(details))
