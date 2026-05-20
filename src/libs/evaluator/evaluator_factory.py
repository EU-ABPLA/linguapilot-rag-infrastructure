from typing import Any, Callable, Dict, List, Mapping, Sequence

from libs.evaluator.base_evaluator import BaseEvaluator, UnavailableEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation.composite_evaluator import CompositeEvaluator
from observability.evaluation.ragas_evaluator import RagasEvaluator

EvaluatorBuilder = Callable[[], BaseEvaluator]


class EvaluatorFactory:
    _registry: Dict[str, EvaluatorBuilder] = {}

    @classmethod
    def register(cls, backend: str, builder: EvaluatorBuilder) -> None:
        normalized = _normalize_backend(backend)
        cls._registry[normalized] = builder

    @classmethod
    def unregister(cls, backend: str) -> None:
        normalized = _normalize_backend(backend)
        cls._registry.pop(normalized, None)

    @classmethod
    def create(cls, settings: Any) -> BaseEvaluator:
        backends = _extract_backends(settings)
        evaluators = [cls._create_single(name) for name in backends]
        if len(evaluators) == 1:
            return evaluators[0]
        return CompositeEvaluator(evaluators)

    @classmethod
    def _create_single(cls, backend: str) -> BaseEvaluator:
        builder = cls._registry.get(backend)
        if builder is None:
            raise ValueError("Unknown evaluator backend: " + backend)
        instance = builder()
        if not isinstance(instance, BaseEvaluator):
            raise TypeError(
                "Factory builder for '" + backend + "' must return BaseEvaluator"
            )
        return instance


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("Invalid evaluator backend")
    return backend.strip().lower().replace("-", "_")


def _extract_backends(settings: Any) -> List[str]:
    values: Any
    if isinstance(settings, Mapping):
        values = _backend_values_from_mapping(settings)
    else:
        values = _backend_values_from_object(settings)
    backends = _normalize_backends(values)
    if not backends:
        raise ValueError("Missing required field: evaluation.backends")
    return backends


def _backend_values_from_mapping(settings: Mapping[str, Any]) -> Any:
    evaluation = settings.get("evaluation")
    if isinstance(evaluation, Mapping):
        if "backends" in evaluation:
            return evaluation["backends"]
        if "backend" in evaluation:
            return [evaluation["backend"]]
        if "provider" in evaluation:
            return [evaluation["provider"]]
    if "backends" in settings:
        return settings["backends"]
    if "backend" in settings:
        return [settings["backend"]]
    if "provider" in settings:
        return [settings["provider"]]
    return []


def _backend_values_from_object(settings: Any) -> Any:
    evaluation = getattr(settings, "evaluation", None)
    if evaluation is not None:
        if hasattr(evaluation, "backends"):
            return getattr(evaluation, "backends")
        if hasattr(evaluation, "backend"):
            return [getattr(evaluation, "backend")]
        if hasattr(evaluation, "provider"):
            return [getattr(evaluation, "provider")]
    if hasattr(settings, "backends"):
        return getattr(settings, "backends")
    if hasattr(settings, "backend"):
        return [getattr(settings, "backend")]
    if hasattr(settings, "provider"):
        return [getattr(settings, "provider")]
    return []


def _normalize_backends(values: Any) -> List[str]:
    if isinstance(values, str):
        return [_normalize_backend(values)]
    if not isinstance(values, Sequence):
        return []
    output: List[str] = []
    for item in values:
        output.append(_normalize_backend(item))
    return output


EvaluatorFactory.register("custom", CustomEvaluator)
EvaluatorFactory.register("ragas", RagasEvaluator)
EvaluatorFactory.register("deepeval", lambda: UnavailableEvaluator("deepeval"))
