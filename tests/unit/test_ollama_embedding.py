from dataclasses import dataclass
from typing import Any, Dict

import pytest

from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.ollama_embedding import OllamaEmbedding


@dataclass
class EmbeddingConfig:
    provider: str


@dataclass
class Settings:
    embedding: EmbeddingConfig


def test_factory_routes_ollama_provider() -> None:
    instance = EmbeddingFactory.create(Settings(embedding=EmbeddingConfig(provider="ollama")))
    assert isinstance(instance, OllamaEmbedding)


def test_ollama_embed_success_with_mock_request() -> None:
    called: Dict[str, Any] = {}

    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        called["url"] = url
        called["headers"] = headers
        called["payload"] = payload
        called["timeout"] = timeout
        return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    embedding = OllamaEmbedding(model="nomic-embed-text", base_url="http://localhost:11434", request_fn=mock_request)
    vectors = embedding.embed(["a", "bb"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert called["url"].endswith("/api/embed")
    assert called["payload"] == {"model": "nomic-embed-text", "input": ["a", "bb"]}


def test_ollama_embed_supports_single_embedding_shape() -> None:
    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        return {"embedding": [1, 2, 3]}

    embedding = OllamaEmbedding(request_fn=mock_request)
    vectors = embedding.embed(["ping"])
    assert vectors == [[1.0, 2.0, 3.0]]


def test_ollama_embed_validation_error_contains_provider() -> None:
    embedding = OllamaEmbedding()
    with pytest.raises(ValueError) as exc_info:
        embedding.embed([])
    assert "ollama validation error" in str(exc_info.value)


def test_ollama_embed_request_error_contains_provider() -> None:
    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        raise RuntimeError("service down")

    embedding = OllamaEmbedding(request_fn=mock_request)
    with pytest.raises(RuntimeError) as exc_info:
        embedding.embed(["ping"])
    assert "ollama request error" in str(exc_info.value)
