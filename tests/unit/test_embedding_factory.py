from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import pytest

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


@dataclass
class EmbeddingConfig:
    provider: str


@dataclass
class Settings:
    embedding: EmbeddingConfig


class FakeEmbedding(BaseEmbedding):
    def embed(
        self, texts: Sequence[str], trace: Optional[Any] = None
    ) -> List[List[float]]:
        return [[float(len(text))] for text in texts]


def test_factory_routes_registered_provider() -> None:
    provider = "fake"
    EmbeddingFactory.register(provider, FakeEmbedding)
    try:
        settings = Settings(embedding=EmbeddingConfig(provider=provider))
        instance = EmbeddingFactory.create(settings)
        assert isinstance(instance, FakeEmbedding)
        assert instance.embed(["a", "bb"]) == [[1.0], [2.0]]
    finally:
        EmbeddingFactory.unregister(provider)


def test_factory_supports_mapping_settings() -> None:
    provider = "mapping-provider"
    EmbeddingFactory.register(provider, FakeEmbedding)
    try:
        settings = {"embedding": {"provider": provider}}
        instance = EmbeddingFactory.create(settings)
        assert isinstance(instance, FakeEmbedding)
    finally:
        EmbeddingFactory.unregister(provider)


def test_factory_raises_for_unknown_provider() -> None:
    settings = Settings(embedding=EmbeddingConfig(provider="unknown-provider"))
    with pytest.raises(ValueError) as exc_info:
        EmbeddingFactory.create(settings)
    assert "Unknown embedding provider: unknown-provider" in str(exc_info.value)
