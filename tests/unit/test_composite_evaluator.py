from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pytest

from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.composite_evaluator import CompositeEvaluator


class _HitEvaluator(BaseEvaluator):
    def evaluate(
        self,
        query: str,
        retrieved_ids: Sequence[str],
        golden_ids: Sequence[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        return {"hit_rate": 1.0}


class _MrrEvaluator(BaseEvaluator):
    def evaluate(
        self,
        query: str,
        retrieved_ids: Sequence[str],
        golden_ids: Sequence[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        return {"mrr": 0.5}


def test_composite_evaluator_merges_metrics() -> None:
    evaluator = CompositeEvaluator([_HitEvaluator(), _MrrEvaluator()])
    metrics = evaluator.evaluate(
        query="how to setup azure openai",
        retrieved_ids=["chunk_1", "chunk_2"],
        golden_ids=["chunk_2"],
    )
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5


def test_composite_evaluator_rejects_empty_list() -> None:
    with pytest.raises(ValueError) as exc_info:
        CompositeEvaluator([])
    assert "non-empty list" in str(exc_info.value)


@dataclass
class _EvaluationConfig:
    backends: List[str]


@dataclass
class _Settings:
    evaluation: _EvaluationConfig


def test_factory_builds_composite_when_multiple_backends() -> None:
    EvaluatorFactory.register("fake_hit", _HitEvaluator)
    EvaluatorFactory.register("fake_mrr", _MrrEvaluator)
    try:
        settings = _Settings(evaluation=_EvaluationConfig(backends=["fake_hit", "fake_mrr"]))
        instance = EvaluatorFactory.create(settings)
        assert isinstance(instance, CompositeEvaluator)
        metrics = instance.evaluate(
            query="query",
            retrieved_ids=["a"],
            golden_ids=["a"],
        )
        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == 0.5
    finally:
        EvaluatorFactory.unregister("fake_hit")
        EvaluatorFactory.unregister("fake_mrr")
