from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder


@dataclass(frozen=True)
class BatchEncodingResult:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    dense_vector: List[float]
    sparse_stats: Dict[str, Any]


class BatchProcessor:
    def __init__(
        self,
        settings: Any,
        dense_encoder: Optional[DenseEncoder] = None,
        sparse_encoder: Optional[SparseEncoder] = None,
        batch_size: Optional[int] = None,
    ):
        self._settings = settings
        self._dense_encoder = dense_encoder or DenseEncoder(settings)
        self._sparse_encoder = sparse_encoder or SparseEncoder()
        self._batch_size = _extract_batch_size(settings, batch_size)
        if self._batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

    def process(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[BatchEncodingResult]:
        if not chunks:
            return []
        output: List[BatchEncodingResult] = []
        for batch_index, start in enumerate(range(0, len(chunks), self._batch_size)):
            batch = chunks[start : start + self._batch_size]
            started_at = time.perf_counter()
            dense_vectors = self._dense_encoder.encode(batch, trace=trace)
            sparse_stats = self._sparse_encoder.encode(batch, trace=trace)
            if len(dense_vectors) != len(batch):
                raise RuntimeError(
                    "batch processor error: dense vector count mismatch, expected "
                    + str(len(batch))
                    + " got "
                    + str(len(dense_vectors))
                )
            if len(sparse_stats) != len(batch):
                raise RuntimeError(
                    "batch processor error: sparse stats count mismatch, expected "
                    + str(len(batch))
                    + " got "
                    + str(len(sparse_stats))
                )
            for index, chunk in enumerate(batch):
                output.append(
                    BatchEncodingResult(
                        chunk_id=chunk.id,
                        text=chunk.text,
                        metadata=dict(chunk.metadata),
                        dense_vector=list(dense_vectors[index]),
                        sparse_stats=dict(sparse_stats[index]),
                    )
                )
            if trace is not None:
                trace.record_stage(
                    "batch_processor",
                    {
                        "status": "ok",
                        "batch_index": batch_index,
                        "start_index": start,
                        "batch_size": len(batch),
                        "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                    },
                )
        return output


def _extract_batch_size(settings: Any, override: Optional[int]) -> int:
    if override is not None:
        return _coerce_batch_size(override)
    if isinstance(settings, Mapping):
        ingestion = settings.get("ingestion")
        if isinstance(ingestion, Mapping):
            batch_processor = ingestion.get("batch_processor")
            if isinstance(batch_processor, Mapping):
                value = batch_processor.get("batch_size")
                if value is not None:
                    return _coerce_batch_size(value)
        return 16
    ingestion = getattr(settings, "ingestion", None)
    if ingestion is None:
        return 16
    batch_processor = getattr(ingestion, "batch_processor", None)
    if batch_processor is None:
        return 16
    value = getattr(batch_processor, "batch_size", 16)
    return _coerce_batch_size(value)


def _coerce_batch_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("batch_size must be an integer")
    return value
