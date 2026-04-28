from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from core.query_engine.sparse_retriever import SparseRetriever
from core.trace.trace_context import TraceContext
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.chroma_store import ChromaStore


class FakeBM25Indexer:
    def __init__(self, hits: Optional[List[Mapping[str, Any]]] = None):
        self.hits = list(hits) if hits is not None else []
        self.query_calls: List[Dict[str, Any]] = []

    def query(self, query: str, top_k: int = 5) -> List[Mapping[str, Any]]:
        self.query_calls.append({"query": query, "top_k": top_k})
        return list(self.hits)


class FakeVectorStore(BaseVectorStore):
    def __init__(self, records: Optional[List[Mapping[str, Any]]] = None):
        self.records = list(records) if records is not None else []
        self.ids_calls: List[List[str]] = []

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
        return []

    def get_by_ids(self, ids: Sequence[str]) -> List[Mapping[str, Any]]:
        lookup = set(ids)
        self.ids_calls.append([str(item) for item in ids])
        return [item for item in self.records if str(item.get("id", "")) in lookup]


def test_sparse_retriever_retrieves_and_merges_scores() -> None:
    bm25 = FakeBM25Indexer(
        [
            {"chunk_id": "chunk-2", "score": 2.4},
            {"chunk_id": "chunk-1", "score": 1.2},
        ]
    )
    store = FakeVectorStore(
        [
            {"id": "chunk-1", "text": "alpha text", "metadata": {"source_path": "a.md"}},
            {"id": "chunk-2", "content": "beta text", "metadata": {"source_path": "b.md"}},
        ]
    )
    retriever = SparseRetriever(
        settings={"retrieval": {"top_k": 3}},
        bm25_indexer=bm25,
        vector_store=store,
    )
    results = retriever.retrieve(["Beta", "alpha"], top_k=2)
    assert [item.chunk_id for item in results] == ["chunk-2", "chunk-1"]
    assert [item.score for item in results] == [2.4, 1.2]
    assert results[0].text == "beta text"
    assert results[1].text == "alpha text"
    assert bm25.query_calls[0] == {"query": "beta alpha", "top_k": 2}
    assert store.ids_calls[0] == ["chunk-2", "chunk-1"]


def test_sparse_retriever_uses_default_top_k_from_settings() -> None:
    bm25 = FakeBM25Indexer([{"chunk_id": "chunk-1", "score": 0.4}])
    store = FakeVectorStore(
        [{"id": "chunk-1", "text": "x", "metadata": {"source_path": "a.md"}}]
    )
    retriever = SparseRetriever(
        settings={"retrieval": {"top_k": 7}},
        bm25_indexer=bm25,
        vector_store=store,
    )
    retriever.retrieve(["x"])
    assert bm25.query_calls[0]["top_k"] == 7


def test_sparse_retriever_records_trace_stage() -> None:
    retriever = SparseRetriever(
        settings={"retrieval": {"top_k": 5}},
        bm25_indexer=FakeBM25Indexer([{"chunk_id": "chunk-1", "score": 0.9}]),
        vector_store=FakeVectorStore(
            [{"id": "chunk-1", "text": "trace", "metadata": {"source_path": "x.md"}}]
        ),
    )
    trace = TraceContext(trace_type="query")
    results = retriever.retrieve(["trace"], trace=trace)
    assert len(results) == 1
    stages = [stage for stage in trace.stages if stage["stage"] == "sparse_retriever"]
    assert len(stages) == 1
    assert stages[0]["details"]["result_count"] == 1


def test_sparse_retriever_rejects_invalid_keywords_type() -> None:
    retriever = SparseRetriever(
        settings={"retrieval": {"top_k": 5}},
        bm25_indexer=FakeBM25Indexer(),
        vector_store=FakeVectorStore(),
    )
    with pytest.raises(ValueError) as exc_info:
        retriever.retrieve("not-a-list")  # type: ignore[arg-type]
    assert "keywords must be a sequence of strings" in str(exc_info.value)


def test_sparse_retriever_hits_expected_chunk_with_real_bm25_and_chroma(
    tmp_path: Path,
) -> None:
    bm25_dir = tmp_path / "bm25"
    chroma_dir = tmp_path / "chroma"
    bm25 = BM25Indexer(index_dir=str(bm25_dir))
    bm25.build(
        [
            {"chunk_id": "c1", "doc_length": 2, "term_weights": {"azure": 2}},
            {"chunk_id": "c2", "doc_length": 2, "term_weights": {"openai": 2}},
        ],
        persist=True,
    )
    store = ChromaStore(persist_directory=str(chroma_dir), collection="sparse")
    store.upsert(
        [
            {
                "id": "c1",
                "vector": [1.0, 0.0],
                "content": "azure setup guide",
                "metadata": {"source_path": "docs/azure.md", "collection": "default"},
            },
            {
                "id": "c2",
                "vector": [0.0, 1.0],
                "content": "openai setup guide",
                "metadata": {"source_path": "docs/openai.md", "collection": "default"},
            },
        ]
    )
    retriever = SparseRetriever(
        settings={"retrieval": {"top_k": 2}},
        bm25_indexer=BM25Indexer(index_dir=str(bm25_dir)),
        vector_store=store,
    )
    results = retriever.retrieve(["azure"], top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].text == "azure setup guide"
    assert results[0].metadata["source_path"] == "docs/azure.md"


def test_chroma_store_get_by_ids_returns_text_and_metadata(tmp_path: Path) -> None:
    store = ChromaStore(persist_directory=str(tmp_path), collection="get_by_ids")
    store.upsert(
        [
            {
                "id": "chunk-1",
                "vector": [1.0, 0.0],
                "content": "first",
                "metadata": {"source_path": "a.md", "collection": "default"},
            },
            {
                "id": "chunk-2",
                "vector": [0.0, 1.0],
                "content": "second",
                "metadata": {"source_path": "b.md", "collection": "default"},
            },
        ]
    )
    records = store.get_by_ids(["chunk-2", "missing", "chunk-1"])
    assert [item["id"] for item in records] == ["chunk-2", "chunk-1"]
    assert records[0]["text"] == "second"
    assert records[1]["metadata"]["source_path"] == "a.md"
