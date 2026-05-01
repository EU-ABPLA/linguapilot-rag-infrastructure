from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.ragas_evaluator import RagasEvaluator


class _FakeDataset:
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload


def test_ragas_evaluator_returns_expected_metrics() -> None:
    captured: Dict[str, Any] = {}

    def _fake_evaluate(dataset: Dict[str, Any], metrics: List[Any]) -> Dict[str, float]:
        captured["dataset"] = dataset
        captured["metrics"] = metrics
        return {
            "faithfulness": 0.91,
            "answer_relevancy": 0.87,
            "context_precision": 0.73,
        }

    evaluator = RagasEvaluator(
        runtime_loader=lambda: {
            "dataset_cls": _FakeDataset,
            "evaluate_fn": _fake_evaluate,
            "faithfulness": object(),
            "answer_relevancy": object(),
            "context_precision": object(),
        }
    )
    metrics = evaluator.evaluate(
        query="how to configure azure openai",
        retrieved_ids=["chunk_a", "chunk_b"],
        golden_ids=["chunk_b"],
    )
    assert metrics["faithfulness"] == 0.91
    assert metrics["answer_relevancy"] == 0.87
    assert metrics["context_precision"] == 0.73
    assert captured["dataset"]["question"] == ["how to configure azure openai"]
    assert captured["dataset"]["contexts"] == [["chunk_a", "chunk_b"]]


def test_ragas_evaluator_raises_clear_import_error_when_runtime_missing() -> None:
    evaluator = RagasEvaluator(
        runtime_loader=lambda: (_ for _ in ()).throw(ImportError("missing ragas"))
    )
    with pytest.raises(ImportError) as exc_info:
        evaluator.evaluate(
            query="what is rag",
            retrieved_ids=["chunk_1"],
            golden_ids=["chunk_1"],
        )
    assert "missing ragas" in str(exc_info.value)


@dataclass
class _EvaluationConfig:
    backends: List[str]


@dataclass
class _Settings:
    evaluation: _EvaluationConfig


def test_factory_routes_ragas_backend() -> None:
    settings = _Settings(evaluation=_EvaluationConfig(backends=["ragas"]))
    instance = EvaluatorFactory.create(settings)
    assert isinstance(instance, RagasEvaluator)
