from dataclasses import dataclass

import pytest

from libs.llm.azure_vision_llm import AzureVisionLLM
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_vision_llm import OpenAICompatibleVisionLLM


@dataclass
class VisionLLMConfig:
    provider: str


@dataclass
class Settings:
    vision_llm: VisionLLMConfig


class FakeVisionLLM(BaseVisionLLM):
    def chat_with_image(self, text, image_path, trace=None) -> str:
        return "ok"


def test_factory_routes_azure_vision_llm_for_default_provider() -> None:
    instance = LLMFactory.create_vision_llm(Settings(vision_llm=VisionLLMConfig(provider="azure")))
    assert isinstance(instance, AzureVisionLLM)


def test_factory_routes_registered_vision_provider() -> None:
    provider = "fake-vision"
    LLMFactory.register_vision(provider, FakeVisionLLM)
    try:
        instance = LLMFactory.create_vision_llm(
            Settings(vision_llm=VisionLLMConfig(provider=provider))
        )
        assert isinstance(instance, FakeVisionLLM)
        assert instance.chat_with_image("t", "img.png") == "ok"
    finally:
        LLMFactory.unregister_vision(provider)


def test_factory_supports_mapping_settings_for_vision_llm() -> None:
    provider = "mapping-vision"
    LLMFactory.register_vision(provider, FakeVisionLLM)
    try:
        instance = LLMFactory.create_vision_llm(
            {"vision_llm": {"provider": provider}}
        )
        assert isinstance(instance, FakeVisionLLM)
    finally:
        LLMFactory.unregister_vision(provider)


def test_factory_falls_back_to_openai_compatible_llm_provider_for_vision() -> None:
    instance = LLMFactory.create_vision_llm({"llm": {"provider": "openai"}})
    assert isinstance(instance, OpenAICompatibleVisionLLM)


def test_factory_routes_openai_vision_with_config() -> None:
    instance = LLMFactory.create_vision_llm(
        {
            "vision_llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "k",
                "base_url": "https://api.openai.com/v1",
            }
        }
    )
    assert isinstance(instance, OpenAICompatibleVisionLLM)
    assert instance.provider_name == "openai"
    assert instance.model == "gpt-4o-mini"


def test_factory_raises_for_missing_vision_provider() -> None:
    with pytest.raises(ValueError) as exc_info:
        LLMFactory.create_vision_llm({})
    assert "Missing required field: vision_llm.provider" in str(exc_info.value)
