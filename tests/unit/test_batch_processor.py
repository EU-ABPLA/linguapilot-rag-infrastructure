from typing import List

import pytest

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.batch_processor import BatchProcessor


class FakeDenseEncoder:
    def __init__(self):
        self.calls: List[List[str]] = []
        self._mismatch = False

    def encode(self, chunks: List[Chunk], trace=None):
        self.calls.append([chunk.id for chunk in chunks])
        if self._mismatch:
            return [[1.0]] * (len(chunks) + 1)
        return [[float(len(chunk.text))] for chunk in chunks]


class FakeSparseEncoder:
    def __init__(self):
        self.calls: List[List[str]] = []
        self._mismatch = False

    def encode(self, chunks: List[Chunk], trace=None):
        self.calls.append([chunk.id for chunk in chunks])
        if self._mismatch:
            return [{"chunk_id": "x", "doc_length": 0, "term_weights": {}}] * (
                len(chunks) + 1
            )
        return [
            {
                "chunk_id": chunk.id,
                "doc_length": len(chunk.text.split()),
                "term_weights": {"token": len(chunk.text.split())},
            }
            for chunk in chunks
        ]


def _chunk(index: int) -> Chunk:
    text = "chunk " + str(index)
    return Chunk(
        id="c" + str(index),
        text=text,
        metadata={"source_path": "data/docs/sample.md"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_batch_processor_splits_five_chunks_into_three_batches_when_size_two() -> None:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    processor = BatchProcessor(
        settings={"ingestion": {"batch_processor": {"batch_size": 2}}},
        dense_encoder=dense,
        sparse_encoder=sparse,
        batch_size=2,
    )
    chunks = [_chunk(1), _chunk(2), _chunk(3), _chunk(4), _chunk(5)]
    results = processor.process(chunks)
    assert [result.chunk_id for result in results] == ["c1", "c2", "c3", "c4", "c5"]
    assert dense.calls == [["c1", "c2"], ["c3", "c4"], ["c5"]]
    assert sparse.calls == [["c1", "c2"], ["c3", "c4"], ["c5"]]


def test_batch_processor_keeps_order_stable() -> None:
    processor = BatchProcessor(
        settings={"ingestion": {"batch_processor": {"batch_size": 2}}},
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2,
    )
    chunks = [_chunk(3), _chunk(1), _chunk(2)]
    results = processor.process(chunks)
    assert [result.chunk_id for result in results] == ["c3", "c1", "c2"]


def test_batch_processor_returns_empty_for_empty_chunks() -> None:
    processor = BatchProcessor(
        settings={},
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2,
    )
    assert processor.process([]) == []


def test_batch_processor_raises_when_dense_count_mismatch() -> None:
    dense = FakeDenseEncoder()
    dense._mismatch = True
    processor = BatchProcessor(
        settings={},
        dense_encoder=dense,
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2,
    )
    with pytest.raises(RuntimeError) as exc_info:
        processor.process([_chunk(1), _chunk(2)])
    assert "dense vector count mismatch" in str(exc_info.value)


def test_batch_processor_raises_when_sparse_count_mismatch() -> None:
    sparse = FakeSparseEncoder()
    sparse._mismatch = True
    processor = BatchProcessor(
        settings={},
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=sparse,
        batch_size=2,
    )
    with pytest.raises(RuntimeError) as exc_info:
        processor.process([_chunk(1), _chunk(2)])
    assert "sparse stats count mismatch" in str(exc_info.value)


def test_batch_processor_records_batch_stages_to_trace() -> None:
    processor = BatchProcessor(
        settings={},
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2,
    )
    trace = TraceContext(trace_type="ingestion")
    processor.process([_chunk(1), _chunk(2), _chunk(3)], trace=trace)
    stages = [stage for stage in trace.stages if stage["stage"] == "batch_processor"]
    assert len(stages) == 2
    assert stages[0]["details"]["batch_index"] == 0
    assert stages[1]["details"]["batch_index"] == 1
