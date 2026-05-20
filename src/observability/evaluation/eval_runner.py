from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.reranker import Reranker
from core.types import RetrievalResult
from libs.evaluator.base_evaluator import BaseEvaluator


@dataclass
class EvalCaseResult:
    query: str
    expected_chunk_ids: List[str]
    expected_sources: List[str]
    retrieved_chunk_ids: List[str]
    retrieved_sources: List[str]
    hit: bool
    mrr: float
    source_hit: bool
    metrics: Dict[str, float]
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "expected_sources": list(self.expected_sources),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "retrieved_sources": list(self.retrieved_sources),
            "hit": self.hit,
            "mrr": self.mrr,
            "source_hit": self.source_hit,
            "metrics": dict(self.metrics),
            "error": self.error,
        }


@dataclass
class EvalReport:
    total_cases: int
    completed_cases: int
    failed_cases: int
    hit_rate: float
    mrr: float
    source_hit_rate: float
    avg_metrics: Dict[str, float]
    cases: List[EvalCaseResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "completed_cases": self.completed_cases,
            "failed_cases": self.failed_cases,
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "source_hit_rate": self.source_hit_rate,
            "avg_metrics": dict(self.avg_metrics),
            "cases": [item.to_dict() for item in self.cases],
        }


class EvalRunner:
    def __init__(
        self,
        settings: Any,
        hybrid_search: HybridSearch,
        evaluator: BaseEvaluator,
        reranker: Optional[Reranker] = None,
    ):
        if not isinstance(hybrid_search, HybridSearch):
            raise ValueError("hybrid_search must be HybridSearch")
        if not isinstance(evaluator, BaseEvaluator):
            raise ValueError("evaluator must be BaseEvaluator")
        self._settings = settings
        self._hybrid_search = hybrid_search
        self._evaluator = evaluator
        self._reranker = reranker or Reranker(settings)
        self._default_top_k = _resolve_top_k(settings)

    def run(self, test_set_path: str) -> EvalReport:
        payload = _load_test_set(test_set_path)
        rows = payload.get("test_cases")
        if not isinstance(rows, list):
            raise ValueError("golden test set must contain test_cases list")
        cases: List[EvalCaseResult] = []
        for item in rows:
            case = _parse_case(item)
            cases.append(self._run_case(case))
        return _aggregate_report(cases)

    def _run_case(self, case: Dict[str, Any]) -> EvalCaseResult:
        query = case["query"]
        expected_chunk_ids = case["expected_chunk_ids"]
        expected_sources = case["expected_sources"]
        top_k = case["top_k"] or self._default_top_k
        filters = case["filters"]
        try:
            candidates = self._hybrid_search.search(
                query,
                top_k=top_k,
                filters=filters,
            )
            ranked = self._reranker.rerank(
                query,
                candidates,
                top_k=top_k,
            )
            retrieved_ids = [item.chunk_id for item in ranked]
            retrieved_sources = [str(item.metadata.get("source_path", "")) for item in ranked]
            hit, mrr = _compute_hit_and_mrr(retrieved_ids, expected_chunk_ids)
            source_hit = _compute_source_hit(retrieved_sources, expected_sources)
            metrics = self._evaluator.evaluate(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=expected_chunk_ids,
            )
            return EvalCaseResult(
                query=query,
                expected_chunk_ids=list(expected_chunk_ids),
                expected_sources=list(expected_sources),
                retrieved_chunk_ids=retrieved_ids,
                retrieved_sources=retrieved_sources,
                hit=hit,
                mrr=mrr,
                source_hit=source_hit,
                metrics=_normalize_metrics(metrics),
            )
        except Exception as exc:
            return EvalCaseResult(
                query=query,
                expected_chunk_ids=list(expected_chunk_ids),
                expected_sources=list(expected_sources),
                retrieved_chunk_ids=[],
                retrieved_sources=[],
                hit=False,
                mrr=0.0,
                source_hit=False,
                metrics={},
                error=str(exc),
            )


def _load_test_set(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise ValueError("golden test set not found: " + path)
    raw = file_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise ValueError("invalid golden test set json") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("golden test set root must be object")
    return dict(parsed)


def _parse_case(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("test_case must be object")
    query = value.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("test_case.query must be non-empty string")
    expected_chunk_ids = _to_str_list(value.get("expected_chunk_ids"), "expected_chunk_ids")
    expected_sources = _to_str_list(value.get("expected_sources"), "expected_sources")
    top_k_value = value.get("top_k")
    top_k = None
    if top_k_value is not None:
        if isinstance(top_k_value, bool) or not isinstance(top_k_value, int) or top_k_value <= 0:
            raise ValueError("test_case.top_k must be positive integer")
        top_k = int(top_k_value)
    filters_raw = value.get("filters", {})
    filters: Dict[str, Any] = {}
    if isinstance(filters_raw, Mapping):
        filters = dict(filters_raw)
    return {
        "query": query.strip(),
        "expected_chunk_ids": expected_chunk_ids,
        "expected_sources": expected_sources,
        "top_k": top_k,
        "filters": filters,
    }


def _to_str_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("test_case." + field_name + " must be list")
    output: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            output.append(item.strip())
    return output


def _compute_hit_and_mrr(retrieved_ids: Sequence[str], expected_ids: Sequence[str]) -> tuple[bool, float]:
    golden = set(expected_ids)
    if not golden:
        return False, 0.0
    for index, item_id in enumerate(retrieved_ids, start=1):
        if item_id in golden:
            return True, 1.0 / float(index)
    return False, 0.0


def _compute_source_hit(retrieved_sources: Sequence[str], expected_sources: Sequence[str]) -> bool:
    if not expected_sources:
        return False
    expected = {item for item in expected_sources if item}
    if not expected:
        return False
    for item in retrieved_sources:
        if item in expected:
            return True
    return False


def _normalize_metrics(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("evaluator metrics must be object")
    output: Dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        output[key.strip()] = float(item)
    return output


def _aggregate_report(cases: Sequence[EvalCaseResult]) -> EvalReport:
    total_cases = len(cases)
    completed_cases = 0
    failed_cases = 0
    hit_count = 0
    mrr_sum = 0.0
    source_hit_count = 0
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    for item in cases:
        if item.error:
            failed_cases += 1
            continue
        completed_cases += 1
        if item.hit:
            hit_count += 1
        if item.source_hit:
            source_hit_count += 1
        mrr_sum += float(item.mrr)
        for key, value in item.metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            metric_counts[key] = metric_counts.get(key, 0) + 1
    denominator = float(completed_cases) if completed_cases > 0 else 1.0
    avg_metrics: Dict[str, float] = {}
    for key, value in metric_sums.items():
        count = metric_counts.get(key, 0)
        if count > 0:
            avg_metrics[key] = value / float(count)
    return EvalReport(
        total_cases=total_cases,
        completed_cases=completed_cases,
        failed_cases=failed_cases,
        hit_rate=float(hit_count) / denominator if completed_cases > 0 else 0.0,
        mrr=mrr_sum / denominator if completed_cases > 0 else 0.0,
        source_hit_rate=float(source_hit_count) / denominator if completed_cases > 0 else 0.0,
        avg_metrics=avg_metrics,
        cases=list(cases),
    )


def _resolve_top_k(settings: Any) -> int:
    retrieval = getattr(settings, "retrieval", None)
    if retrieval is not None:
        value = getattr(retrieval, "top_k", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    if isinstance(settings, Mapping):
        retrieval_data = settings.get("retrieval")
        if isinstance(retrieval_data, Mapping):
            value = retrieval_data.get("top_k")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return 5
