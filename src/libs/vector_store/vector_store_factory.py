from typing import Any, Callable, Dict, Mapping

from libs.vector_store.base_vector_store import BaseVectorStore, UnavailableVectorStore

VectorStoreBuilder = Callable[[], BaseVectorStore]


class VectorStoreFactory:
	_registry: Dict[str, VectorStoreBuilder] = {}

	@classmethod
	def register(cls, provider: str, builder: VectorStoreBuilder) -> None:
		normalized = _normalize_provider(provider)
		cls._registry[normalized] = builder

	@classmethod
	def unregister(cls, provider: str) -> None:
		normalized = _normalize_provider(provider)
		cls._registry.pop(normalized, None)

	@classmethod
	def create(cls, settings: Any) -> BaseVectorStore:
		provider = _extract_provider(settings)
		builder = cls._registry.get(provider)
		if builder is None:
			raise ValueError("Unknown vector_store provider: " + provider)
		instance = builder()
		if not isinstance(instance, BaseVectorStore):
			raise TypeError(
				"Factory builder for '" + provider + "' must return BaseVectorStore"
			)
		return instance


def _normalize_provider(provider: str) -> str:
	if not isinstance(provider, str) or not provider.strip():
		raise ValueError("Invalid vector_store provider")
	return provider.strip().lower().replace("-", "_")


def _extract_provider(settings: Any) -> str:
	provider: Any
	if isinstance(settings, Mapping):
		provider = _provider_from_mapping(settings)
	else:
		provider = _provider_from_object(settings)
	return _normalize_provider(provider)


def _provider_from_mapping(settings: Mapping[str, Any]) -> Any:
	vector_store = settings.get("vector_store")
	if isinstance(vector_store, Mapping) and "provider" in vector_store:
		return vector_store["provider"]
	if "provider" in settings:
		return settings["provider"]
	raise ValueError("Missing required field: vector_store.provider")


def _provider_from_object(settings: Any) -> Any:
	vector_store = getattr(settings, "vector_store", None)
	if vector_store is not None and hasattr(vector_store, "provider"):
		return getattr(vector_store, "provider")
	if hasattr(settings, "provider"):
		return getattr(settings, "provider")
	raise ValueError("Missing required field: vector_store.provider")


for _provider in ("chroma",):
	VectorStoreFactory.register(
		_provider, lambda provider=_provider: UnavailableVectorStore(provider)
	)
