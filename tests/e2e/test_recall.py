from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import ProcessedQuery
from core.query_engine.reranker import Reranker
from core.types import RetrievalResult
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation.eval_runner import EvalRunner

MIN_HIT_AT_K = 0.66


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
        rows = _rows_for_query(query)
        if isinstance(filters, Mapping) and filters:
            output = []
            for item in rows:
                if all(item.metadata.get(key) == value for key, value in filters.items()):
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
        return []


class _FakeFusion:
    def fuse(
        self,
        dense: Sequence[RetrievalResult],
        sparse: Sequence[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        merged = list(dense) + list(sparse)
        if isinstance(top_k, int) and top_k > 0:
            return merged[:top_k]
        return merged


def _rows_for_query(query: str) -> List[RetrievalResult]:
    value = query.lower()
    if "azure openai endpoint" in value:
        return [
            RetrievalResult(
                chunk_id="chunk-azure-guide",
                score=0.99,
                text="azure endpoint setup",
                metadata={"source_path": "docs/azure_guide.md", "collection": "cloud"},
            ),
            RetrievalResult(
                chunk_id="chunk-openai-guide",
                score=0.6,
                text="openai setup",
                metadata={"source_path": "docs/openai_guide.md", "collection": "ai"},
            ),
        ]
    if "openai setup guide" in value:
        return [
            RetrievalResult(
                chunk_id="chunk-openai-guide",
                score=0.98,
                text="openai guide",
                metadata={"source_path": "docs/openai_guide.md", "collection": "ai"},
            ),
            RetrievalResult(
                chunk_id="chunk-azure-guide",
                score=0.5,
                text="azure setup",
                metadata={"source_path": "docs/azure_guide.md", "collection": "cloud"},
            ),
        ]
    if "azure troubleshooting faq" in value:
        return [
            RetrievalResult(
                chunk_id="chunk-azure-guide",
                score=0.97,
                text="azure guide",
                metadata={"source_path": "docs/azure_guide.md", "collection": "cloud"},
            ),
            RetrievalResult(
                chunk_id="chunk-openai-guide",
                score=0.7,
                text="openai guide",
                metadata={"source_path": "docs/openai_guide.md", "collection": "ai"},
            ),
        ]
    return []


def _build_hybrid() -> HybridSearch:
    settings = {
        "retrieval": {"top_k": 5},
        "embedding": {"provider": "fake"},
    }
    return HybridSearch(
        settings=settings,
        query_processor=_FakeQueryProcessor(),
        dense_retriever=_FakeDenseRetriever(),
        sparse_retriever=_FakeSparseRetriever(),
        fusion=_FakeFusion(),
    )


def test_recall_hit_at_k_meets_min_threshold() -> None:
    root = Path(__file__).resolve().parents[2]
    golden_path = root / "tests" / "fixtures" / "golden_test_set.json"
    settings = {
        "retrieval": {"top_k": 5},
        "rerank": {"enabled": False, "provider": "none"},
    }
    runner = EvalRunner(
        settings=settings,
        hybrid_search=_build_hybrid(),
        evaluator=CustomEvaluator(),
        reranker=Reranker(settings=settings),
    )
    report = runner.run(str(golden_path))
    assert report.total_cases >= 3
    assert report.completed_cases == report.total_cases
    assert report.hit_rate >= MIN_HIT_AT_K
