from typing import Any, List, Optional, Sequence

import pytest

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.dense_encoder import DenseEncoder
from libs.embedding.base_embedding import BaseEmbedding


class FakeEmbedding(BaseEmbedding):
    def __init__(self, vectors: Optional[List[List[float]]] = None, error: Optional[Exception] = None):
        self.vectors = vectors if vectors is not None else [[0.1, 0.2]]
        self.error = error
        self.calls: List[Sequence[str]] = []

    def embed(
        self, texts: Sequence[str], trace: Optional[Any] = None
    ) -> List[List[float]]:
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        if len(self.vectors) == len(texts):
            return [list(row) for row in self.vectors]
        if len(self.vectors) == 1:
            return [list(self.vectors[0]) for _ in texts]
        return [list(row) for row in self.vectors]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "data/docs/sample.md"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_dense_encoder_encodes_chunks_in_order() -> None:
    embedding = FakeEmbedding(vectors=[[1.0, 0.0], [0.0, 1.0]])
    encoder = DenseEncoder(settings={"embedding": {"provider": "fake"}}, embedding=embedding)
    chunks = [_chunk("c1", "alpha"), _chunk("c2", "beta")]
    vectors = encoder.encode(chunks)
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert embedding.calls[0] == ["alpha", "beta"]


def test_dense_encoder_returns_empty_for_empty_chunks() -> None:
    embedding = FakeEmbedding()
    encoder = DenseEncoder(settings={"embedding": {"provider": "fake"}}, embedding=embedding)
    vectors = encoder.encode([])
    assert vectors == []
    assert embedding.calls == []


def test_dense_encoder_raises_when_vector_count_mismatch() -> None:
    embedding = FakeEmbedding(vectors=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    encoder = DenseEncoder(settings={"embedding": {"provider": "fake"}}, embedding=embedding)
    chunks = [_chunk("c1", "a"), _chunk("c2", "b")]
    with pytest.raises(RuntimeError) as exc_info:
        encoder.encode(chunks)
    assert "vector count mismatch" in str(exc_info.value)


def test_dense_encoder_raises_when_vector_dimension_inconsistent() -> None:
    embedding = FakeEmbedding(vectors=[[1.0], [2.0, 3.0]])
    encoder = DenseEncoder(settings={"embedding": {"provider": "fake"}}, embedding=embedding)
    chunks = [_chunk("c1", "a"), _chunk("c2", "b")]
    with pytest.raises(RuntimeError) as exc_info:
        encoder.encode(chunks)
    assert "inconsistent embedding dimensions" in str(exc_info.value)


def test_dense_encoder_raises_when_vector_is_empty() -> None:
    embedding = FakeEmbedding(vectors=[[]])
    encoder = DenseEncoder(settings={"embedding": {"provider": "fake"}}, embedding=embedding)
    chunks = [_chunk("c1", "a")]
    with pytest.raises(RuntimeError) as exc_info:
        encoder.encode(chunks)
    assert "vector must be non-empty" in str(exc_info.value)


def test_dense_encoder_propagates_embedding_errors() -> None:
    embedding = FakeEmbedding(error=RuntimeError("embedding failed"))
    encoder = DenseEncoder(settings={"embedding": {"provider": "fake"}}, embedding=embedding)
    chunks = [_chunk("c1", "a")]
    with pytest.raises(RuntimeError) as exc_info:
        encoder.encode(chunks)
    assert "embedding failed" in str(exc_info.value)


def test_dense_encoder_records_trace_stage() -> None:
    embedding = FakeEmbedding(vectors=[[0.1, 0.2], [0.3, 0.4]])
    encoder = DenseEncoder(settings={"embedding": {"provider": "fake"}}, embedding=embedding)
    trace = TraceContext(trace_type="ingestion")
    chunks = [_chunk("c1", "hello"), _chunk("c2", "world")]
    vectors = encoder.encode(chunks, trace=trace)
    assert len(vectors) == 2
    assert any(stage["stage"] == "dense_encoder" for stage in trace.stages)
