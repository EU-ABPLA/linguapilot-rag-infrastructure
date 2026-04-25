from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


@dataclass
class SplitterConfig:
    provider: str


@dataclass
class Settings:
    splitter: SplitterConfig


class FakeSplitter(BaseSplitter):
    def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
        if not text:
            return []
        return text.split(" ")


def test_factory_routes_registered_provider() -> None:
    provider = "fake"
    SplitterFactory.register(provider, FakeSplitter)
    try:
        settings = Settings(splitter=SplitterConfig(provider=provider))
        instance = SplitterFactory.create(settings)
        assert isinstance(instance, FakeSplitter)
        assert instance.split_text("a bb ccc") == ["a", "bb", "ccc"]
    finally:
        SplitterFactory.unregister(provider)


def test_factory_supports_mapping_settings() -> None:
    provider = "mapping-provider"
    SplitterFactory.register(provider, FakeSplitter)
    try:
        settings = {"splitter": {"provider": provider}}
        instance = SplitterFactory.create(settings)
        assert isinstance(instance, FakeSplitter)
    finally:
        SplitterFactory.unregister(provider)


def test_factory_raises_for_unknown_provider() -> None:
    settings = Settings(splitter=SplitterConfig(provider="unknown-provider"))
    with pytest.raises(ValueError) as exc_info:
        SplitterFactory.create(settings)
    assert "Unknown splitter provider: unknown_provider" in str(exc_info.value)
