from typing import Any, Callable, Dict, Mapping

from libs.llm.azure_llm import AzureLLM
from libs.llm.azure_vision_llm import AzureVisionLLM
from libs.llm.base_llm import BaseLLM
from libs.llm.base_vision_llm import BaseVisionLLM, UnavailableVisionLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.ollama_llm import OllamaLLM
from libs.llm.openai_llm import OpenAILLM
from libs.llm.openai_vision_llm import OpenAICompatibleVisionLLM

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
        if provider == "openai":
            return OpenAILLM(
                model=_extract_llm_value(settings, "model", "gpt-4o-mini"),
                api_key=_extract_llm_value(settings, "api_key", ""),
                base_url=_extract_llm_value(settings, "base_url", "https://api.openai.com/v1"),
            )
        if provider == "deepseek":
            return DeepSeekLLM(
                model=_extract_llm_value(settings, "model", "deepseek-chat"),
                api_key=_extract_llm_value(settings, "api_key", ""),
                base_url=_extract_llm_value(settings, "base_url", "https://api.deepseek.com/v1"),
            )
        if provider == "azure":
            return AzureLLM(
                model=_extract_llm_value(settings, "model", "gpt-4o-mini"),
                api_key=_extract_llm_value(settings, "api_key", ""),
                endpoint=_extract_llm_value(settings, "base_url", "https://example.openai.azure.com"),
            )
        if provider == "ollama":
            return OllamaLLM(
                model=_extract_llm_value(settings, "model", "llama3.1"),
                base_url=_extract_llm_value(settings, "base_url", "http://localhost:11434"),
            )
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
        if provider == "azure":
            return _build_azure_vision_llm(settings)
        if provider in ("openai", "openai-compatible", "openai_compatible"):
            return _build_openai_compatible_vision_llm(provider, settings)
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


def _extract_llm_value(settings: Any, field: str, default: str) -> str:
    if isinstance(settings, Mapping):
        llm = settings.get("llm")
        if isinstance(llm, Mapping):
            value = llm.get(field, default)
        else:
            value = settings.get(field, default)
    else:
        llm = getattr(settings, "llm", None)
        if llm is not None:
            value = getattr(llm, field, default)
        else:
            value = getattr(settings, field, default)
    if isinstance(value, str):
        return value
    return default


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


def _build_azure_vision_llm(settings: Any) -> BaseVisionLLM:
    return AzureVisionLLM(
        deployment_name=_extract_vision_value(settings, "deployment_name", _extract_vision_value(settings, "model", "gpt-4o")),
        api_key=_extract_vision_value(settings, "api_key", ""),
        endpoint=_extract_vision_value(settings, "endpoint", _extract_vision_value(settings, "base_url", "https://example.openai.azure.com")),
        api_version=_extract_vision_value(settings, "api_version", "2024-02-15-preview"),
        max_image_size=_extract_vision_int(settings, "max_image_size", 2048),
    )


def _build_openai_compatible_vision_llm(provider: str, settings: Any) -> BaseVisionLLM:
    provider_name = "openai" if provider in ("openai-compatible", "openai_compatible") else provider
    return OpenAICompatibleVisionLLM(
        provider_name=provider_name,
        model=_extract_vision_value(settings, "model", "gpt-4o-mini"),
        api_key=_extract_vision_value(settings, "api_key", ""),
        base_url=_extract_vision_value(settings, "base_url", _default_vision_base_url(provider_name)),
        max_image_size=_extract_vision_int(settings, "max_image_size", 2048),
    )


def _default_vision_base_url(provider: str) -> str:
    return "https://api.openai.com/v1"


def _extract_vision_value(settings: Any, field: str, default: str) -> str:
    vision = _extract_vision_config(settings)
    if isinstance(vision, Mapping):
        value = vision.get(field, default)
    else:
        value = getattr(vision, field, default)
    if isinstance(value, str):
        return value
    return default


def _extract_vision_int(settings: Any, field: str, default: int) -> int:
    vision = _extract_vision_config(settings)
    if isinstance(vision, Mapping):
        value = vision.get(field, default)
    else:
        value = getattr(vision, field, default)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _extract_vision_config(settings: Any) -> Any:
    if isinstance(settings, Mapping):
        vision = settings.get("vision_llm")
        if isinstance(vision, Mapping):
            return vision
        return {}
    vision = getattr(settings, "vision_llm", None)
    if vision is not None:
        return vision
    return object()


LLMFactory.register("openai", OpenAILLM)
LLMFactory.register("azure", AzureLLM)
LLMFactory.register("deepseek", DeepSeekLLM)
LLMFactory.register("ollama", OllamaLLM)

for _provider in ("ollama",):
    LLMFactory.register_vision(
        _provider, lambda provider=_provider: UnavailableVisionLLM(provider)
    )

LLMFactory.register_vision("azure", AzureVisionLLM)
