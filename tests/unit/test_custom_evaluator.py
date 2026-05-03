from dataclasses import dataclass
from typing import List

import pytest

from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


@dataclass
class EvaluationConfig:
    backends: List[str]


@dataclass
class Settings:
    evaluation: EvaluationConfig


def test_custom_evaluator_metrics_hit_and_mrr() -> None:
    evaluator = CustomEvaluator()
    metrics = evaluator.evaluate(
        query="what is rag",
        retrieved_ids=["a", "b", "c"],
        golden_ids=["b", "x"],
    )
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5


def test_custom_evaluator_metrics_miss() -> None:
    evaluator = CustomEvaluator()
    metrics = evaluator.evaluate(
        query="what is rag",
        retrieved_ids=["a", "b", "c"],
        golden_ids=["x", "y"],
    )
    assert metrics["hit_rate"] == 0.0
    assert metrics["mrr"] == 0.0


def test_factory_routes_backend_from_evaluation_backends() -> None:
    settings = Settings(evaluation=EvaluationConfig(backends=["custom"]))
    instance = EvaluatorFactory.create(settings)
    assert isinstance(instance, CustomEvaluator)


def test_factory_supports_mapping_settings() -> None:
    settings = {"evaluation": {"backends": ["custom"]}}
    instance = EvaluatorFactory.create(settings)
    assert isinstance(instance, CustomEvaluator)


def test_factory_raises_for_unknown_backend() -> None:
    settings = Settings(evaluation=EvaluationConfig(backends=["unknown-backend"]))
    with pytest.raises(ValueError) as exc_info:
        EvaluatorFactory.create(settings)
    assert "Unknown evaluator backend: unknown_backend" in str(exc_info.value)


def test_custom_evaluator_empty_golden_ids_is_miss() -> None:
    evaluator = CustomEvaluator()
    metrics = evaluator.evaluate(
        query="what is rag",
        retrieved_ids=["a", "b", "c"],
        golden_ids=[],
    )
    assert metrics["hit_rate"] == 0.0
    assert metrics["mrr"] == 0.0


def test_custom_evaluator_prefers_first_hit_for_mrr() -> None:
    evaluator = CustomEvaluator()
    metrics = evaluator.evaluate(
        query="ranking",
        retrieved_ids=["x", "y", "x", "z"],
        golden_ids=["x"],
    )
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0


def test_custom_evaluator_handles_duplicate_golden_ids() -> None:
    evaluator = CustomEvaluator()
    metrics = evaluator.evaluate(
        query="ranking",
        retrieved_ids=["m", "n", "o"],
        golden_ids=["o", "o", "o"],
    )
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0 / 3.0
