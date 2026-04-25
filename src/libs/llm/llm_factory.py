from typing import Any, Callable, Dict, Mapping

from libs.llm.azure_llm import AzureLLM
from libs.llm.azure_vision_llm import AzureVisionLLM
from libs.llm.base_llm import BaseLLM
from libs.llm.base_vision_llm import BaseVisionLLM, UnavailableVisionLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.ollama_llm import OllamaLLM
from libs.llm.openai_llm import OpenAILLM

LLMBuilder = Callable[[], BaseLLM]
VisionLLMBuilder = Callable[[], BaseVisionLLM]


class LLMFactory:
    _registry: Dict[str, LLMBuilder] = {}
    _vision_registry: Dict[str, VisionLLMBuilder] = {}

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

    @classmethod
    def register_vision(cls, provider: str, builder: VisionLLMBuilder) -> None:
        normalized = _normalize_provider(provider)
        cls._vision_registry[normalized] = builder

    @classmethod
    def unregister_vision(cls, provider: str) -> None:
        normalized = _normalize_provider(provider)
        cls._vision_registry.pop(normalized, None)

    @classmethod
    def create_vision_llm(cls, settings: Any) -> BaseVisionLLM:
        provider = _extract_vision_provider(settings)
        builder = cls._vision_registry.get(provider)
        if builder is None:
            raise ValueError(f"Unknown Vision LLM provider: {provider}")
        instance = builder()
        if not isinstance(instance, BaseVisionLLM):
            raise TypeError(
                f"Factory builder for '{provider}' must return BaseVisionLLM"
            )
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


def _extract_vision_provider(settings: Any) -> str:
    provider: Any
    if isinstance(settings, Mapping):
        provider = _vision_provider_from_mapping(settings)
    else:
        provider = _vision_provider_from_object(settings)
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


def _vision_provider_from_mapping(settings: Mapping[str, Any]) -> Any:
    vision_llm = settings.get("vision_llm")
    if isinstance(vision_llm, Mapping) and "provider" in vision_llm:
        return vision_llm["provider"]
    if isinstance(vision_llm, Mapping) and "backend" in vision_llm:
        return vision_llm["backend"]
    llm = settings.get("llm")
    if isinstance(llm, Mapping) and "provider" in llm:
        return llm["provider"]
    if "provider" in settings:
        return settings["provider"]
    raise ValueError("Missing required field: vision_llm.provider")


def _vision_provider_from_object(settings: Any) -> Any:
    vision_llm = getattr(settings, "vision_llm", None)
    if vision_llm is not None and hasattr(vision_llm, "provider"):
        return getattr(vision_llm, "provider")
    if vision_llm is not None and hasattr(vision_llm, "backend"):
        return getattr(vision_llm, "backend")
    llm = getattr(settings, "llm", None)
    if llm is not None and hasattr(llm, "provider"):
        return getattr(llm, "provider")
    if hasattr(settings, "provider"):
        return getattr(settings, "provider")
    raise ValueError("Missing required field: vision_llm.provider")


LLMFactory.register("openai", OpenAILLM)
LLMFactory.register("azure", AzureLLM)
LLMFactory.register("deepseek", DeepSeekLLM)
LLMFactory.register("ollama", OllamaLLM)

for _provider in ("openai", "azure", "deepseek", "ollama"):
    LLMFactory.register_vision(
        _provider, lambda provider=_provider: UnavailableVisionLLM(provider)
    )

LLMFactory.register_vision("azure", AzureVisionLLM)
