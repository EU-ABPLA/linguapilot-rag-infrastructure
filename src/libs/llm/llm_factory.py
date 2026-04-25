from typing import Any, Callable, Dict, Mapping

from libs.llm.azure_llm import AzureLLM
from libs.llm.base_llm import BaseLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.ollama_llm import OllamaLLM
from libs.llm.openai_llm import OpenAILLM

LLMBuilder = Callable[[], BaseLLM]


class LLMFactory:
    _registry: Dict[str, LLMBuilder] = {}

    @classmethod
    def register(cls, provider: str, builder: LLMBuilder) -> None:
        normalized = _normalize_provider(provider)
        cls._registry[normalized] = builder

    @classmethod
    def unregister(cls, provider: str) -> None:
        normalized = _normalize_provider(provider)
        cls._registry.pop(normalized, None)

    @classmethod
    def create(cls, settings: Any) -> BaseLLM:
        provider = _extract_provider(settings)
        builder = cls._registry.get(provider)
        if builder is None:
            raise ValueError(f"Unknown LLM provider: {provider}")
        instance = builder()
        if not isinstance(instance, BaseLLM):
            raise TypeError(f"Factory builder for '{provider}' must return BaseLLM")
        return instance


def _normalize_provider(provider: str) -> str:
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Invalid LLM provider")
    return provider.strip().lower()


def _extract_provider(settings: Any) -> str:
    provider: Any
    if isinstance(settings, Mapping):
        provider = _provider_from_mapping(settings)
    else:
        provider = _provider_from_object(settings)
    return _normalize_provider(provider)


def _provider_from_mapping(settings: Mapping[str, Any]) -> Any:
    llm = settings.get("llm")
    if isinstance(llm, Mapping) and "provider" in llm:
        return llm["provider"]
    if "provider" in settings:
        return settings["provider"]
    raise ValueError("Missing required field: llm.provider")


def _provider_from_object(settings: Any) -> Any:
    llm = getattr(settings, "llm", None)
    if llm is not None and hasattr(llm, "provider"):
        return getattr(llm, "provider")
    if hasattr(settings, "provider"):
        return getattr(settings, "provider")
    raise ValueError("Missing required field: llm.provider")


LLMFactory.register("openai", OpenAILLM)
LLMFactory.register("azure", AzureLLM)
LLMFactory.register("deepseek", DeepSeekLLM)
LLMFactory.register("ollama", OllamaLLM)
