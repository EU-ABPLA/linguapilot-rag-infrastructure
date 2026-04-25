from typing import Any, Callable, Dict, Mapping

from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.ollama_embedding import OllamaEmbedding
from libs.embedding.openai_embedding import OpenAIEmbedding

EmbeddingBuilder = Callable[[], BaseEmbedding]


class EmbeddingFactory:
	_registry: Dict[str, EmbeddingBuilder] = {}

	@classmethod
	def register(cls, provider: str, builder: EmbeddingBuilder) -> None:
		normalized = _normalize_provider(provider)
		cls._registry[normalized] = builder

	@classmethod
	def unregister(cls, provider: str) -> None:
		normalized = _normalize_provider(provider)
		cls._registry.pop(normalized, None)

	@classmethod
	def create(cls, settings: Any) -> BaseEmbedding:
		provider = _extract_provider(settings)
		builder = cls._registry.get(provider)
		if builder is None:
			raise ValueError(f"Unknown embedding provider: {provider}")
		instance = builder()
		if not isinstance(instance, BaseEmbedding):
			raise TypeError(f"Factory builder for '{provider}' must return BaseEmbedding")
		return instance


def _normalize_provider(provider: str) -> str:
	if not isinstance(provider, str) or not provider.strip():
		raise ValueError("Invalid embedding provider")
	return provider.strip().lower()


def _extract_provider(settings: Any) -> str:
	provider: Any
	if isinstance(settings, Mapping):
		provider = _provider_from_mapping(settings)
	else:
		provider = _provider_from_object(settings)
	return _normalize_provider(provider)


def _provider_from_mapping(settings: Mapping[str, Any]) -> Any:
	embedding = settings.get("embedding")
	if isinstance(embedding, Mapping) and "provider" in embedding:
		return embedding["provider"]
	if "provider" in settings:
		return settings["provider"]
	raise ValueError("Missing required field: embedding.provider")


def _provider_from_object(settings: Any) -> Any:
	embedding = getattr(settings, "embedding", None)
	if embedding is not None and hasattr(embedding, "provider"):
		return getattr(embedding, "provider")
	if hasattr(settings, "provider"):
		return getattr(settings, "provider")
	raise ValueError("Missing required field: embedding.provider")


EmbeddingFactory.register("openai", OpenAIEmbedding)
EmbeddingFactory.register("azure", AzureEmbedding)
EmbeddingFactory.register("ollama", OllamaEmbedding)
