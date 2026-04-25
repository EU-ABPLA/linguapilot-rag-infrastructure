from dataclasses import dataclass
from typing import Any, Dict

import pytest

from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAIEmbedding


@dataclass
class EmbeddingConfig:
    provider: str


@dataclass
class Settings:
    embedding: EmbeddingConfig


def test_factory_routes_openai_provider() -> None:
    instance = EmbeddingFactory.create(Settings(embedding=EmbeddingConfig(provider="openai")))
    assert isinstance(instance, OpenAIEmbedding)


def test_factory_routes_azure_provider() -> None:
    instance = EmbeddingFactory.create(Settings(embedding=EmbeddingConfig(provider="azure")))
    assert isinstance(instance, AzureEmbedding)


def test_openai_embed_validation_error_contains_provider() -> None:
    embedding = OpenAIEmbedding(api_key="k")
    with pytest.raises(ValueError) as exc_info:
        embedding.embed([])
    assert "openai validation error" in str(exc_info.value)


def test_openai_embed_success_with_mock_request() -> None:
    called: Dict[str, Any] = {}

    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        called["url"] = url
        called["headers"] = headers
        called["payload"] = payload
        called["timeout"] = timeout
        return {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}

    embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="k", request_fn=mock_request)
    vectors = embedding.embed(["a", "bb"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert called["url"].endswith("/embeddings")
    assert called["payload"]["model"] == "text-embedding-3-small"
    assert called["payload"]["input"] == ["a", "bb"]
    assert called["headers"]["Authorization"] == "Bearer k"


def test_azure_embed_success_with_mock_request() -> None:
    called: Dict[str, Any] = {}

    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        called["url"] = url
        called["headers"] = headers
        called["payload"] = payload
        return {"data": [{"embedding": [1, 2, 3]}]}

    embedding = AzureEmbedding(
        model="embed-deployment",
        api_key="azure-key",
        endpoint="https://example.openai.azure.com",
        api_version="2024-02-15-preview",
        request_fn=mock_request,
    )
    vectors = embedding.embed(["ping"])
    assert vectors == [[1.0, 2.0, 3.0]]
    assert "deployments/embed-deployment/embeddings" in called["url"]
    assert "api-version=2024-02-15-preview" in called["url"]
    assert called["headers"]["api-key"] == "azure-key"
    assert called["payload"] == {"input": ["ping"]}


def test_azure_embed_request_error_contains_provider() -> None:
    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        raise RuntimeError("network down")

    embedding = AzureEmbedding(model="m", api_key="k", request_fn=mock_request)
    with pytest.raises(RuntimeError) as exc_info:
        embedding.embed(["ping"])
    assert "azure request error" in str(exc_info.value)
