from typing import Any, Callable, Dict, Mapping

from libs.reranker.base_reranker import BaseReranker, NoneReranker
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.llm_reranker import LLMReranker

RerankerBuilder = Callable[[], BaseReranker]


class RerankerFactory:
	_registry: Dict[str, RerankerBuilder] = {}

	@classmethod
	def register(cls, backend: str, builder: RerankerBuilder) -> None:
		normalized = _normalize_backend(backend)
		cls._registry[normalized] = builder

	@classmethod
	def unregister(cls, backend: str) -> None:
		normalized = _normalize_backend(backend)
		cls._registry.pop(normalized, None)

	@classmethod
	def create(cls, settings: Any) -> BaseReranker:
		backend = _extract_backend(settings)
		builder = cls._registry.get(backend)
		if builder is None:
			raise ValueError("Unknown reranker backend: " + backend)
		instance = builder()
		if not isinstance(instance, BaseReranker):
			raise TypeError(
				"Factory builder for '" + backend + "' must return BaseReranker"
			)
		return instance


def _normalize_backend(backend: str) -> str:
	if not isinstance(backend, str) or not backend.strip():
		raise ValueError("Invalid reranker backend")
	return backend.strip().lower().replace("-", "_")


def _extract_backend(settings: Any) -> str:
	backend: Any
	if isinstance(settings, Mapping):
		backend = _backend_from_mapping(settings)
	else:
		backend = _backend_from_object(settings)
	return _normalize_backend(backend)


def _backend_from_mapping(settings: Mapping[str, Any]) -> Any:
	rerank = settings.get("rerank")
	if isinstance(rerank, Mapping) and "provider" in rerank:
		return rerank["provider"]
	if isinstance(rerank, Mapping) and "backend" in rerank:
		return rerank["backend"]
	if "backend" in settings:
		return settings["backend"]
	if "provider" in settings:
		return settings["provider"]
	raise ValueError("Missing required field: rerank.provider")


def _backend_from_object(settings: Any) -> Any:
	rerank = getattr(settings, "rerank", None)
	if rerank is not None and hasattr(rerank, "provider"):
		return getattr(rerank, "provider")
	if rerank is not None and hasattr(rerank, "backend"):
		return getattr(rerank, "backend")
	if hasattr(settings, "provider"):
		return getattr(settings, "provider")
	if hasattr(settings, "backend"):
		return getattr(settings, "backend")
	raise ValueError("Missing required field: rerank.provider")


RerankerFactory.register("none", NoneReranker)
RerankerFactory.register("llm", LLMReranker)
RerankerFactory.register("cross_encoder", CrossEncoderReranker)
