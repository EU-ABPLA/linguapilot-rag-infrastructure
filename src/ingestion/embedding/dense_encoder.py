from __future__ import annotations

from typing import Any, List, Optional

from core.trace.trace_context import TraceContext
from core.types import Chunk
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class DenseEncoder:
    def __init__(self, settings: Any, embedding: Optional[BaseEmbedding] = None):
        self._settings = settings
        self._embedding = embedding or EmbeddingFactory.create(settings)

    def encode(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[List[float]]:
        if not chunks:
            if trace is not None:
                trace.record_stage(
                    "dense_encoder",
                    {"status": "ok", "chunk_count": 0, "vector_count": 0},
                )
            return []
        texts = [chunk.text for chunk in chunks]
        vectors = self._embedding.embed(texts, trace=trace)
        if len(vectors) != len(chunks):
            raise RuntimeError(
                "dense encoder error: vector count mismatch, expected "
                + str(len(chunks))
                + " got "
                + str(len(vectors))
            )
        dimension = len(vectors[0]) if vectors else 0
        for vector in vectors:
            if not vector:
                raise RuntimeError("dense encoder error: embedding vector must be non-empty")
            if len(vector) != dimension:
                raise RuntimeError("dense encoder error: inconsistent embedding dimensions")
        if trace is not None:
            trace.record_stage(
                "dense_encoder",
                {
                    "status": "ok",
                    "chunk_count": len(chunks),
                    "vector_count": len(vectors),
                    "dimension": dimension,
                },
            )
        return vectors
