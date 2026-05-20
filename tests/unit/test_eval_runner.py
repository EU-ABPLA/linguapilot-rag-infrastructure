from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import ProcessedQuery
from core.query_engine.reranker import Reranker
from core.types import RetrievalResult
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation.eval_runner import EvalRunner


class _FakeQueryProcessor:
    def process(
        self,
        query: str,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> ProcessedQuery:
        output_filters: Dict[str, Any] = {}
        if isinstance(filters, Mapping):
            output_filters = dict(filters)
        return ProcessedQuery(
            query=query,
            keywords=[query.lower()],
            filters=output_filters,
        )


class _FakeDenseRetriever:
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        if "boom" in query.lower():
            raise RuntimeError("dense route error")
        rows = [
            RetrievalResult(
                chunk_id="chunk-a",
                score=0.9,
                text="alpha",
                metadata={"source_path": "docs/a.md", "collection": "default"},
            ),
            RetrievalResult(
                chunk_id="chunk-b",
                score=0.8,
                text="beta",
                metadata={"source_path": "docs/b.md", "collection": "default"},
            ),
        ]
        if isinstance(filters, Mapping) and filters:
            output = []
            for item in rows:
                if all(item.metadata.get(k) == v for k, v in filters.items()):
                    output.append(item)
            rows = output
        if isinstance(top_k, int) and top_k > 0:
            return rows[:top_k]
        return rows


class _FakeSparseRetriever:
    def retrieve(
        self,
        keywords: Sequence[str],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        first = keywords[0] if keywords else ""
        if "boom" in first.lower():
            raise RuntimeError("sparse route error")
        return []


class _FakeFusion:
    def fuse(
        self,
        dense: Sequence[RetrievalResult],
        sparse: Sequence[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        rows = list(dense) + list(sparse)
        if isinstance(top_k, int) and top_k > 0:
            return rows[:top_k]
        return rows


def _build_hybrid() -> HybridSearch:
    settings = {
        "retrieval": {"top_k": 3},
        "embedding": {"provider": "fake"},
    }
    return HybridSearch(
        settings=settings,
        query_processor=_FakeQueryProcessor(),
        dense_retriever=_FakeDenseRetriever(),
        sparse_retriever=_FakeSparseRetriever(),
        fusion=_FakeFusion(),
    )


def _build_runner() -> EvalRunner:
    settings = {
        "retrieval": {"top_k": 3},
        "rerank": {"enabled": False, "provider": "none"},
    }
    return EvalRunner(
        settings=settings,
        hybrid_search=_build_hybrid(),
        evaluator=CustomEvaluator(),
        reranker=Reranker(settings=settings),
    )


def test_eval_runner_builds_report_with_hit_and_mrr(tmp_path: Path) -> None:
    test_set = tmp_path / "golden.json"
    test_set.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "find beta",
                        "expected_chunk_ids": ["chunk-b"],
                        "expected_sources": ["docs/b.md"],
                        "top_k": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = _build_runner().run(str(test_set))
    assert report.total_cases == 1
    assert report.completed_cases == 1
    assert report.failed_cases == 0
    assert report.hit_rate == 1.0
    assert report.mrr == 0.5
    assert report.source_hit_rate == 1.0
    assert report.avg_metrics["hit_rate"] == 1.0
    assert report.avg_metrics["mrr"] == 0.5


def test_eval_runner_continues_when_case_fails(tmp_path: Path) -> None:
    test_set = tmp_path / "golden.json"
    test_set.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "boom case",
                        "expected_chunk_ids": ["chunk-a"],
                        "expected_sources": ["docs/a.md"],
                    },
                    {
                        "query": "normal case",
                        "expected_chunk_ids": ["chunk-a"],
                        "expected_sources": ["docs/a.md"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    report = _build_runner().run(str(test_set))
    assert report.total_cases == 2
    assert report.completed_cases == 1
    assert report.failed_cases == 1
    assert report.hit_rate == 1.0
    assert report.cases[0].error != ""
    assert report.cases[1].error == ""
