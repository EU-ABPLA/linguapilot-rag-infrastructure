from dataclasses import dataclass
from typing import Mapping, Sequence

import pytest

from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


@dataclass
class LLMConfig:
    provider: str


@dataclass
class Settings:
    llm: LLMConfig


class FakeLLM(BaseLLM):
    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        return "ok"


def test_factory_routes_registered_provider() -> None:
    provider = "fake"
    LLMFactory.register(provider, FakeLLM)
    try:
        settings = Settings(llm=LLMConfig(provider=provider))
        instance = LLMFactory.create(settings)
        assert isinstance(instance, FakeLLM)
        assert instance.chat([{"role": "user", "content": "ping"}]) == "ok"
    finally:
        LLMFactory.unregister(provider)


def test_factory_supports_mapping_settings() -> None:
    provider = "mapping-provider"
    LLMFactory.register(provider, FakeLLM)
    try:
        settings = {"llm": {"provider": provider}}
        instance = LLMFactory.create(settings)
        assert isinstance(instance, FakeLLM)
    finally:
        LLMFactory.unregister(provider)


def test_factory_raises_for_unknown_provider() -> None:
    settings = Settings(llm=LLMConfig(provider="unknown-provider"))
    with pytest.raises(ValueError) as exc_info:
        LLMFactory.create(settings)
    assert "Unknown LLM provider: unknown-provider" in str(exc_info.value)
