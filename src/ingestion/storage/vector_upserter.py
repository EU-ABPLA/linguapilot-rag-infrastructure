from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.trace.trace_context import TraceContext
from core.types import Chunk
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserter:
    def __init__(
        self,
        settings: Any,
        vector_store: Optional[BaseVectorStore] = None,
        collection: Optional[str] = None,
    ):
        self._settings = settings
        self._vector_store = vector_store or VectorStoreFactory.create(settings)
        self._collection = _extract_collection(settings, collection)

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[Sequence[float]],
        trace: Optional[TraceContext] = None,
    ) -> List[str]:
        if len(chunks) != len(dense_vectors):
            raise RuntimeError(
                "vector upserter error: chunk/vector count mismatch, expected "
                + str(len(chunks))
                + " got "
                + str(len(dense_vectors))
            )
        if not chunks:
            if trace is not None:
                trace.record_stage(
                    "vector_upserter",
                    {
                        "status": "ok",
                        "upsert_count": 0,
                        "collection": self._collection,
                    },
                )
            return []
        records: List[Dict[str, Any]] = []
        ids: List[str] = []
        for index, chunk in enumerate(chunks):
            vector = _normalize_vector(dense_vectors[index])
            chunk_index = _extract_chunk_index(chunk, index)
            source_path = str(chunk.metadata["source_path"])
            chunk_id = _build_chunk_id(source_path, chunk_index, chunk.text)
            metadata = dict(chunk.metadata)
            metadata["collection"] = self._collection
            metadata["chunk_id"] = chunk.id
            if chunk.source_ref is not None:
                metadata["source_ref"] = chunk.source_ref
            records.append(
                {
                    "id": chunk_id,
                    "vector": vector,
                    "content": chunk.text,
                    "metadata": metadata,
                }
            )
            ids.append(chunk_id)
        self._vector_store.upsert(records, trace=trace)
        if trace is not None:
            trace.record_stage(
                "vector_upserter",
                {
                    "status": "ok",
                    "upsert_count": len(records),
                    "collection": self._collection,
                },
            )
        return ids


def _extract_collection(settings: Any, override: Optional[str]) -> str:
    if override is not None:
        if not isinstance(override, str) or not override.strip():
            raise ValueError("collection must be non-empty")
        return override.strip()
    if isinstance(settings, Mapping):
        vector_store = settings.get("vector_store")
        if isinstance(vector_store, Mapping):
            raw = vector_store.get("collection")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return "default"
    vector_store = getattr(settings, "vector_store", None)
    if vector_store is None:
        return "default"
    raw = getattr(vector_store, "collection", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "default"


def _extract_chunk_index(chunk: Chunk, fallback: int) -> int:
    raw = chunk.metadata.get("chunk_index")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return fallback
    return raw


def _normalize_vector(vector: Sequence[float]) -> List[float]:
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise ValueError("vector must be a sequence")
    values: List[float] = []
    for item in vector:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("vector must contain numeric values")
        values.append(float(item))
    if not values:
        raise ValueError("vector must not be empty")
    return values


def _build_chunk_id(source_path: str, chunk_index: int, text: str) -> str:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    payload = source_path + "\n" + str(chunk_index) + "\n" + content_hash[:8]
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
