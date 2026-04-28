from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.storage.vector_upserter import VectorUpserter
from libs.vector_store.base_vector_store import BaseVectorStore


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Any]] = {}
        self.calls: List[List[str]] = []

    def upsert(
        self, records: Sequence[Mapping[str, Any]], trace: Optional[Any] = None
    ) -> None:
        call_ids: List[str] = []
        for record in records:
            record_id = str(record.get("id", ""))
            normalized = {
                "id": record_id,
                "vector": list(record.get("vector", [])),
                "content": str(record.get("content", "")),
                "metadata": dict(record.get("metadata", {})),
            }
            self.records[record_id] = normalized
            call_ids.append(record_id)
        self.calls.append(call_ids)

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[Mapping[str, Any]]:
        return []

    def get_by_ids(self, ids: Sequence[str]) -> List[Mapping[str, Any]]:
        output: List[Mapping[str, Any]] = []
        for item_id in ids:
            record = self.records.get(str(item_id))
            if record is not None:
                output.append(record)
        return output


def _chunk(index: int, text: str, source_path: str = "data/docs/sample.md") -> Chunk:
    return Chunk(
        id="chunk-" + str(index),
        text=text,
        metadata={"source_path": source_path, "chunk_index": index},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_vector_upserter_same_chunk_twice_uses_same_stable_id() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(
        settings={"vector_store": {"provider": "chroma", "collection": "lesson-1"}},
        vector_store=store,
    )
    chunk = _chunk(0, "hello world")
    first_ids = upserter.upsert([chunk], [[0.1, 0.2]])
    second_ids = upserter.upsert([chunk], [[0.1, 0.2]])
    assert first_ids == second_ids
    assert len(store.records) == 1
    stored = store.records[first_ids[0]]
    assert stored["metadata"]["collection"] == "lesson-1"
    assert stored["metadata"]["chunk_id"] == "chunk-0"


def test_vector_upserter_content_change_changes_id() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(
        settings={"vector_store": {"provider": "chroma"}},
        vector_store=store,
    )
    id_a = upserter.upsert([_chunk(0, "alpha")], [[0.3]])[0]
    id_b = upserter.upsert([_chunk(0, "alpha changed")], [[0.3]])[0]
    assert id_a != id_b


def test_vector_upserter_batch_upsert_preserves_order() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(
        settings={"vector_store": {"provider": "chroma"}},
        vector_store=store,
    )
    chunks = [_chunk(0, "first"), _chunk(1, "second"), _chunk(2, "third")]
    ids = upserter.upsert(chunks, [[1.0], [2.0], [3.0]])
    assert ids == store.calls[0]
    assert len(ids) == 3


def test_vector_upserter_raises_on_count_mismatch() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(
        settings={"vector_store": {"provider": "chroma"}},
        vector_store=store,
    )
    with pytest.raises(RuntimeError) as exc_info:
        upserter.upsert([_chunk(0, "x"), _chunk(1, "y")], [[1.0]])
    assert "chunk/vector count mismatch" in str(exc_info.value)


def test_vector_upserter_records_trace_stage() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(
        settings={"vector_store": {"provider": "chroma"}},
        vector_store=store,
    )
    trace = TraceContext(trace_type="ingestion")
    upserter.upsert([_chunk(0, "trace me")], [[0.4, 0.6]], trace=trace)
    stages = [stage for stage in trace.stages if stage["stage"] == "vector_upserter"]
    assert len(stages) == 1
    assert stages[0]["details"]["upsert_count"] == 1
    assert stages[0]["details"]["collection"] == "default"
