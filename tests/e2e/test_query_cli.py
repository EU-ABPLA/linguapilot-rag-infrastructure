from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.types import RetrievalResult


def _load_query_module() -> object:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "query.py"
    spec = importlib.util.spec_from_file_location("query_script_for_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load query.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _DummySettings:
    retrieval: Any
    rerank: Any
    vector_store: Any


class _DummyQueryProcessor:
    @dataclass
    class _Processed:
        query: str
        keywords: List[str]
        filters: Dict[str, Any]

    def process(
        self,
        query: str,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> "_DummyQueryProcessor._Processed":
        tokens = [part.lower() for part in query.split() if part.strip()]
        return self._Processed(query=query, keywords=tokens, filters=dict(filters or {}))


class _DummyDenseRetriever:
    def __init__(self, settings: Any):
        self._settings = settings

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="dense-1",
                score=0.8,
                text="dense result text",
                metadata={"source_path": "docs/dense.md", "collection": "default"},
            )
        ]


class _DummySparseRetriever:
    def __init__(self, settings: Any, bm25_indexer: Optional[Any] = None):
        self._settings = settings
        self._bm25_indexer = bm25_indexer

    def retrieve(
        self,
        keywords: Sequence[str],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="sparse-1",
                score=0.7,
                text="sparse result text",
                metadata={"source_path": "docs/sparse.md", "collection": "default"},
            )
        ]


class _DummyFusion:
    def __init__(self, settings: Any):
        self._settings = settings

    def fuse(
        self,
        dense_results: Sequence[RetrievalResult],
        sparse_results: Sequence[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        output = list(dense_results) + list(sparse_results)
        if top_k is not None:
            output = output[:top_k]
        return output


class _DummyHybridSearch:
    def __init__(
        self,
        settings: Any,
        query_processor: Optional[Any] = None,
        dense_retriever: Optional[Any] = None,
        sparse_retriever: Optional[Any] = None,
        fusion: Optional[Any] = None,
    ):
        self._settings = settings

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        output = [
            RetrievalResult(
                chunk_id="h-1",
                score=0.9,
                text="hybrid first",
                metadata={"source_path": "docs/h1.md", "collection": "default"},
            ),
            RetrievalResult(
                chunk_id="h-2",
                score=0.8,
                text="hybrid second",
                metadata={"source_path": "docs/h2.md", "collection": "default"},
            ),
        ]
        if filters and "collection" in filters:
            output = [item for item in output if item.metadata.get("collection") == filters["collection"]]
        if top_k is not None:
            output = output[:top_k]
        return output

    def _apply_metadata_filters(
        self,
        candidates: List[RetrievalResult],
        filters: Mapping[str, Any],
    ) -> List[RetrievalResult]:
        if not filters:
            return list(candidates)
        return [item for item in candidates if item.metadata.get("collection") == filters.get("collection")]


class _DummyReranker:
    def __init__(self, settings: Any):
        self._settings = settings

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        output = list(reversed(candidates))
        if top_k is not None:
            output = output[:top_k]
        return output


class _DummyBM25Indexer:
    def __init__(self, index_dir: str = "data/db/bm25"):
        self.index_dir = index_dir


def _patch_module(module: object, monkeypatch) -> None:
    settings = _DummySettings(
        retrieval=type("Retrieval", (), {"top_k": 5})(),
        rerank=type("Rerank", (), {"enabled": True, "provider": "none"})(),
        vector_store=type("VectorStore", (), {"provider": "chroma", "collection": "default"})(),
    )
    monkeypatch.setattr(module, "load_settings", lambda path: settings)
    monkeypatch.setattr(module, "QueryProcessor", _DummyQueryProcessor)
    monkeypatch.setattr(module, "DenseRetriever", _DummyDenseRetriever)
    monkeypatch.setattr(module, "SparseRetriever", _DummySparseRetriever)
    monkeypatch.setattr(module, "Fusion", _DummyFusion)
    monkeypatch.setattr(module, "HybridSearch", _DummyHybridSearch)
    monkeypatch.setattr(module, "Reranker", _DummyReranker)
    monkeypatch.setattr(module, "BM25Indexer", _DummyBM25Indexer)


def test_query_cli_default_mode_formats_results(monkeypatch, capsys) -> None:
    module = _load_query_module()
    _patch_module(module, monkeypatch)
    exit_code = module.main(["--query", "test question", "--top-k", "2"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "score=" in output
    assert "source=" in output
    assert "hybrid second" in output
    assert "hybrid first" in output


def test_query_cli_no_rerank_keeps_hybrid_order(monkeypatch, capsys) -> None:
    module = _load_query_module()
    _patch_module(module, monkeypatch)
    exit_code = module.main(["--query", "test question", "--top-k", "2", "--no-rerank"])
    output = capsys.readouterr().out
    assert exit_code == 0
    lines = [line for line in output.splitlines() if line.strip().startswith("1.")]
    assert lines
    assert "hybrid first" in lines[0]


def test_query_cli_verbose_prints_stage_sections(monkeypatch, capsys) -> None:
    module = _load_query_module()
    _patch_module(module, monkeypatch)
    exit_code = module.main(["--query", "test question", "--verbose", "--top-k", "2"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "=== Dense ===" in output
    assert "=== Sparse ===" in output
    assert "=== Fusion ===" in output
    assert "=== Rerank ===" in output


def test_query_cli_prints_friendly_message_when_empty(monkeypatch, capsys) -> None:
    module = _load_query_module()
    _patch_module(module, monkeypatch)

    class _EmptyHybrid:
        def __init__(self, settings: Any, **kwargs: Any):
            self._settings = settings

        def search(
            self,
            query: str,
            top_k: Optional[int] = None,
            filters: Optional[Mapping[str, Any]] = None,
            trace: Optional[Any] = None,
        ) -> List[RetrievalResult]:
            return []

        def _apply_metadata_filters(
            self, candidates: List[RetrievalResult], filters: Mapping[str, Any]
        ) -> List[RetrievalResult]:
            return []

    monkeypatch.setattr(module, "HybridSearch", _EmptyHybrid)
    exit_code = module.main(["--query", "test question", "--top-k", "2"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "未找到相关文档" in output
