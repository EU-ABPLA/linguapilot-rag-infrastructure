from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from libs.reranker.base_reranker import BaseReranker, NoneReranker
from libs.reranker.reranker_factory import RerankerFactory


@dataclass
class RerankConfig:
    provider: str


@dataclass
class Settings:
    rerank: RerankConfig


class FakeReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Any = None,
    ) -> List[Dict[str, Any]]:
        return list(reversed(candidates))


def test_factory_returns_none_reranker_for_none_backend() -> None:
    settings = Settings(rerank=RerankConfig(provider="none"))
    instance = RerankerFactory.create(settings)
    assert isinstance(instance, NoneReranker)


def test_none_reranker_keeps_original_order() -> None:
    settings = {"rerank": {"provider": "none"}}
    instance = RerankerFactory.create(settings)
    candidates = [
        {"id": "a", "score": 0.3},
        {"id": "b", "score": 0.2},
    ]
    result = instance.rerank("query", candidates)
    assert [item["id"] for item in result] == ["a", "b"]


def test_factory_routes_registered_backend() -> None:
    backend = "fake"
    RerankerFactory.register(backend, FakeReranker)
    try:
        settings = Settings(rerank=RerankConfig(provider=backend))
        instance = RerankerFactory.create(settings)
        assert isinstance(instance, FakeReranker)
    finally:
        RerankerFactory.unregister(backend)


def test_factory_raises_for_unknown_backend() -> None:
    settings = Settings(rerank=RerankConfig(provider="unknown-backend"))
    with pytest.raises(ValueError) as exc_info:
        RerankerFactory.create(settings)
    assert "Unknown reranker backend: unknown_backend" in str(exc_info.value)


def test_factory_normalizes_hyphen_backend_name() -> None:
    settings = {"rerank": {"provider": "cross-encoder"}}
    instance = RerankerFactory.create(settings)
    from libs.reranker.cross_encoder_reranker import CrossEncoderReranker

    assert isinstance(instance, CrossEncoderReranker)


def test_factory_raises_for_missing_provider_field() -> None:
    with pytest.raises(ValueError) as exc_info:
        RerankerFactory.create({"rerank": {}})
    assert "Missing required field: rerank.provider" in str(exc_info.value)


def test_factory_raises_for_blank_provider() -> None:
    settings = {"rerank": {"provider": "   "}}
    with pytest.raises(ValueError) as exc_info:
        RerankerFactory.create(settings)
    assert "Invalid reranker backend" in str(exc_info.value)
