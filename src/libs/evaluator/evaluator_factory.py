from typing import Any, Callable, Dict, Mapping

from libs.evaluator.base_evaluator import BaseEvaluator, UnavailableEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator

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
		backend = _extract_backend(settings)
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


def _extract_backend(settings: Any) -> str:
	backend: Any
	if isinstance(settings, Mapping):
		backend = _backend_from_mapping(settings)
	else:
		backend = _backend_from_object(settings)
	return _normalize_backend(backend)


def _backend_from_mapping(settings: Mapping[str, Any]) -> Any:
	evaluation = settings.get("evaluation")
	if isinstance(evaluation, Mapping):
		if "backend" in evaluation:
			return evaluation["backend"]
		if "provider" in evaluation:
			return evaluation["provider"]
		backends = evaluation.get("backends")
		if isinstance(backends, list) and backends:
			return backends[0]
	if "backend" in settings:
		return settings["backend"]
	if "provider" in settings:
		return settings["provider"]
	backends = settings.get("backends")
	if isinstance(backends, list) and backends:
		return backends[0]
	raise ValueError("Missing required field: evaluation.backends")


def _backend_from_object(settings: Any) -> Any:
	evaluation = getattr(settings, "evaluation", None)
	if evaluation is not None and hasattr(evaluation, "backend"):
		return getattr(evaluation, "backend")
	if evaluation is not None and hasattr(evaluation, "provider"):
		return getattr(evaluation, "provider")
	if evaluation is not None and hasattr(evaluation, "backends"):
		backends = getattr(evaluation, "backends")
		if isinstance(backends, list) and backends:
			return backends[0]
	if hasattr(settings, "backend"):
		return getattr(settings, "backend")
	if hasattr(settings, "provider"):
		return getattr(settings, "provider")
	if hasattr(settings, "backends"):
		backends = getattr(settings, "backends")
		if isinstance(backends, list) and backends:
			return backends[0]
	raise ValueError("Missing required field: evaluation.backends")


EvaluatorFactory.register("custom", CustomEvaluator)

for _backend in ("ragas", "deepeval"):
	EvaluatorFactory.register(
		_backend, lambda backend=_backend: UnavailableEvaluator(backend)
	)
