from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class SparseRetriever:
    def __init__(
        self,
        settings: Any,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        self._settings = settings
        self._bm25_indexer = bm25_indexer or _create_bm25_indexer(settings)
        if hasattr(self._bm25_indexer, "load"):
            self._bm25_indexer.load()
        self._vector_store = vector_store or VectorStoreFactory.create(settings)
        self._default_top_k = _extract_top_k(settings)

    def retrieve(
        self,
        keywords: Sequence[str],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        normalized_keywords = _normalize_keywords(keywords)
        if not normalized_keywords:
            return []
        resolved_top_k = _resolve_top_k(self._default_top_k, top_k)
        bm25_query = " ".join(normalized_keywords)
        bm25_hits = self._bm25_indexer.query(bm25_query, top_k=resolved_top_k)
        scored_ids: List[tuple[str, float]] = []
        for item in bm25_hits:
            if not isinstance(item, Mapping):
                continue
            chunk_id = _extract_chunk_id(item)
            if chunk_id is None:
                continue
            scored_ids.append((chunk_id, _extract_score(item)))
        if not scored_ids:
            if trace is not None:
                trace.record_stage(
                    "sparse_retriever",
                    {
                        "status": "ok",
                        "top_k": resolved_top_k,
                        "keyword_count": len(normalized_keywords),
                        "result_count": 0,
                    },
                )
            return []
        chunk_ids = [item[0] for item in scored_ids]
        records = self._vector_store.get_by_ids(chunk_ids)
        record_by_id: Dict[str, Mapping[str, Any]] = {}
        for item in records:
            if not isinstance(item, Mapping):
                continue
            chunk_id = _extract_chunk_id(item)
            if chunk_id is None:
                continue
            if chunk_id not in record_by_id:
                record_by_id[chunk_id] = item
        results: List[RetrievalResult] = []
        for chunk_id, score in scored_ids:
            item = record_by_id.get(chunk_id)
            if item is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=_extract_text(item),
                    metadata=_extract_metadata(item),
                )
            )
        if trace is not None:
            trace.record_stage(
                "sparse_retriever",
                {
                    "status": "ok",
                    "top_k": resolved_top_k,
                    "keyword_count": len(normalized_keywords),
                    "result_count": len(results),
                },
            )
        return results


def _create_bm25_indexer(settings: Any) -> BM25Indexer:
    index_dir = _extract_bm25_index_dir(settings)
    if index_dir is None:
        return BM25Indexer()
    return BM25Indexer(index_dir=index_dir)


def _extract_bm25_index_dir(settings: Any) -> Optional[str]:
    if isinstance(settings, Mapping):
        retrieval = settings.get("retrieval")
        if isinstance(retrieval, Mapping):
            bm25 = retrieval.get("bm25")
            if isinstance(bm25, Mapping):
                candidate = bm25.get("index_dir")
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            candidate = retrieval.get("bm25_index_dir")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        bm25 = settings.get("bm25")
        if isinstance(bm25, Mapping):
            candidate = bm25.get("index_dir")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        candidate = settings.get("bm25_index_dir")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return None
    retrieval = getattr(settings, "retrieval", None)
    if retrieval is not None:
        bm25 = getattr(retrieval, "bm25", None)
        if bm25 is not None:
            candidate = getattr(bm25, "index_dir", None)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        candidate = getattr(retrieval, "bm25_index_dir", None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    bm25 = getattr(settings, "bm25", None)
    if bm25 is not None:
        candidate = getattr(bm25, "index_dir", None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    candidate = getattr(settings, "bm25_index_dir", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return None


def _normalize_keywords(keywords: Sequence[str]) -> List[str]:
    if isinstance(keywords, (str, bytes)) or not isinstance(keywords, Sequence):
        raise ValueError("keywords must be a sequence of strings")
    output: List[str] = []
    for item in keywords:
        if not isinstance(item, str):
            raise ValueError("keywords must be a sequence of strings")
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized not in output:
            output.append(normalized)
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


def _extract_chunk_id(item: Mapping[str, Any]) -> Optional[str]:
    value = item.get("chunk_id", item.get("id"))
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _extract_score(item: Mapping[str, Any]) -> float:
    value = item.get("score", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _extract_text(item: Mapping[str, Any]) -> str:
    value = item.get("text")
    if isinstance(value, str):
        return value
    alt = item.get("content")
    if isinstance(alt, str):
        return alt
    return ""


def _extract_metadata(item: Mapping[str, Any]) -> Dict[str, Any]:
    value = item.get("metadata", {})
    if isinstance(value, Mapping):
        return dict(value)
    return {}
