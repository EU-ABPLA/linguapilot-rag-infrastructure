import math
from pathlib import Path

import pytest

from ingestion.storage.bm25_indexer import BM25Indexer


def _stats(chunk_id: str, doc_length: int, term_weights: dict) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_length": doc_length,
        "term_weights": term_weights,
    }


def test_bm25_build_save_load_query_stable(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=str(tmp_path))
    sparse_stats = [
        _stats("c1", 3, {"alpha": 2, "beta": 1}),
        _stats("c2", 2, {"beta": 1, "gamma": 1}),
        _stats("c3", 2, {"gamma": 1, "delta": 1}),
    ]
    indexer.build(sparse_stats, persist=True)
    first = indexer.query("alpha beta", top_k=3)
    assert len(first) >= 1
    assert first[0]["chunk_id"] == "c1"

    reloaded = BM25Indexer(index_dir=str(tmp_path))
    reloaded.load()
    second = reloaded.query("alpha beta", top_k=3)
    assert [item["chunk_id"] for item in first] == [item["chunk_id"] for item in second]
    assert [round(item["score"], 8) for item in first] == [
        round(item["score"], 8) for item in second
    ]


def test_bm25_idf_matches_formula(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=str(tmp_path))
    sparse_stats = [
        _stats("c1", 2, {"alpha": 1}),
        _stats("c2", 2, {"alpha": 1}),
        _stats("c3", 2, {"beta": 1}),
    ]
    payload = indexer.build(sparse_stats, persist=False)
    actual = payload["index"]["alpha"]["idf"]
    expected = math.log((3 - 2 + 0.5) / (2 + 0.5))
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_bm25_supports_incremental_update_and_rebuild(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=str(tmp_path))
    indexer.build([_stats("c1", 2, {"alpha": 1})], persist=False)
    assert indexer.query("beta", top_k=5) == []

    indexer.update([_stats("c2", 3, {"beta": 2})], persist=False)
    after_update = indexer.query("beta", top_k=5)
    assert after_update[0]["chunk_id"] == "c2"

    indexer.build([_stats("c3", 1, {"gamma": 1})], persist=False)
    assert indexer.query("beta", top_k=5) == []
    assert indexer.query("gamma", top_k=1)[0]["chunk_id"] == "c3"


def test_bm25_query_rejects_invalid_top_k(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=str(tmp_path))
    indexer.build([_stats("c1", 1, {"alpha": 1})], persist=False)
    with pytest.raises(ValueError) as exc_info:
        indexer.query("alpha", top_k=0)
    assert "top_k must be positive" in str(exc_info.value)


def test_bm25_load_missing_file_returns_empty_snapshot(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=str(tmp_path))
    payload = indexer.load()
    assert payload["doc_count"] == 0
    assert payload["index"] == {}


def test_bm25_query_sorts_ties_by_chunk_id(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=str(tmp_path))
    indexer.build(
        [
            _stats("a", 1, {"x": 1}),
            _stats("b", 1, {"x": 1}),
            _stats("c", 1, {"x": 1}),
        ],
        persist=False,
    )
    results = indexer.query("x", top_k=3)
    assert [item["chunk_id"] for item in results] == ["a", "b", "c"]
