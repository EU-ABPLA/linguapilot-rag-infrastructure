from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from core.query_engine.dense_retriever import DenseRetriever
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.embedding.base_embedding import BaseEmbedding
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.chroma_store import ChromaStore


class FakeEmbedding(BaseEmbedding):
    def __init__(self, vectors: Optional[List[List[float]]] = None):
        self.vectors = vectors if vectors is not None else [[0.1, 0.2, 0.3]]
        self.calls: List[List[str]] = []

    def embed(
        self, texts: Sequence[str], trace: Optional[Any] = None
    ) -> List[List[float]]:
        self.calls.append(list(texts))
        return [list(item) for item in self.vectors]


class FakeVectorStore(BaseVectorStore):
    def __init__(self, results: Optional[List[Mapping[str, Any]]] = None):
        self.results = list(results) if results is not None else []
        self.query_calls: List[Dict[str, Any]] = []

    def upsert(
        self, records: Sequence[Mapping[str, Any]], trace: Optional[Any] = None
    ) -> None:
        return

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[Mapping[str, Any]]:
        self.query_calls.append(
            {"vector": list(vector), "top_k": top_k, "filters": dict(filters or {})}
        )
        return list(self.results)

    def get_by_ids(self, ids: Sequence[str]) -> List[Mapping[str, Any]]:
        lookup = set(ids)
        return [item for item in self.results if str(item.get("id", "")) in lookup]


def test_retrieval_result_supports_json_roundtrip() -> None:
    result = RetrievalResult(
        chunk_id="c-1",
        score=0.91,
        text="hello",
        metadata={"source_path": "data/docs/a.md", "collection": "default"},
    )
    restored = RetrievalResult.from_json(result.to_json())
    assert restored == result


def test_dense_retriever_retrieves_and_normalizes_results() -> None:
    embedding = FakeEmbedding([[0.8, 0.2]])
    store = FakeVectorStore(
        [
            {
                "id": "chunk-1",
                "score": 0.95,
                "text": "alpha text",
                "metadata": {"source_path": "data/docs/a.md"},
            },
            {
                "id": "chunk-2",
                "score": 0.77,
                "content": "beta text",
                "metadata": {"source_path": "data/docs/b.md"},
            },
        ]
    )
    retriever = DenseRetriever(
        settings={"retrieval": {"top_k": 3}},
        embedding_client=embedding,
        vector_store=store,
    )
    results = retriever.retrieve("alpha query", top_k=2, filters={"collection": "default"})
    assert [item.chunk_id for item in results] == ["chunk-1", "chunk-2"]
    assert results[0].text == "alpha text"
    assert results[1].text == "beta text"
    assert embedding.calls == [["alpha query"]]
    assert store.query_calls[0]["top_k"] == 2
    assert store.query_calls[0]["filters"] == {"collection": "default"}


def test_dense_retriever_uses_default_top_k_from_settings() -> None:
    embedding = FakeEmbedding([[0.2, 0.9]])
    store = FakeVectorStore(
        [
            {
                "id": "chunk-1",
                "score": 0.1,
                "text": "x",
                "metadata": {"source_path": "data/docs/a.md"},
            }
        ]
    )
    retriever = DenseRetriever(
        settings={"retrieval": {"top_k": 7}},
        embedding_client=embedding,
        vector_store=store,
    )
    retriever.retrieve("fallback topk")
    assert store.query_calls[0]["top_k"] == 7


def test_dense_retriever_raises_when_embedding_result_count_invalid() -> None:
    retriever = DenseRetriever(
        settings={"retrieval": {"top_k": 5}},
        embedding_client=FakeEmbedding([[0.1], [0.2]]),
        vector_store=FakeVectorStore(),
    )
    with pytest.raises(RuntimeError) as exc_info:
        retriever.retrieve("bad embedding count")
    assert "embedding result mismatch" in str(exc_info.value)


def test_dense_retriever_raises_for_invalid_filters_type() -> None:
    retriever = DenseRetriever(
        settings={"retrieval": {"top_k": 5}},
        embedding_client=FakeEmbedding(),
        vector_store=FakeVectorStore(),
    )
    with pytest.raises(ValueError) as exc_info:
        retriever.retrieve("x", filters=["bad"])  # type: ignore[arg-type]
    assert "filters must be a mapping" in str(exc_info.value)


def test_dense_retriever_records_trace_stage() -> None:
    retriever = DenseRetriever(
        settings={"retrieval": {"top_k": 5}},
        embedding_client=FakeEmbedding(),
        vector_store=FakeVectorStore(
            [
                {
                    "id": "chunk-1",
                    "score": 0.9,
                    "text": "trace text",
                    "metadata": {"source_path": "data/docs/a.md"},
                }
            ]
        ),
    )
    trace = TraceContext(trace_type="query")
    retriever.retrieve("trace query", trace=trace)
    stages = [stage for stage in trace.stages if stage["stage"] == "dense_retriever"]
    assert len(stages) == 1
    assert stages[0]["details"]["result_count"] == 1


def test_chroma_store_query_returns_text_field(tmp_path: Path) -> None:
    store = ChromaStore(persist_directory=str(tmp_path), collection="dense_text")
    store.upsert(
        [
            {
                "id": "chunk-1",
                "vector": [1.0, 0.0],
                "content": "stored content",
                "metadata": {"source_path": "data/docs/a.md"},
            }
        ]
    )
    results = store.query([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "stored content"
