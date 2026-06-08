from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import pytest

from libs.llm.azure_llm import AzureLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import OllamaLLM
from libs.llm.openai_llm import OpenAILLM


@dataclass
class LLMConfig:
    provider: str


@dataclass
class Settings:
    llm: LLMConfig


def test_factory_routes_openai_provider() -> None:
    instance = LLMFactory.create(Settings(llm=LLMConfig(provider="openai")))
    assert isinstance(instance, OpenAILLM)


def test_factory_routes_azure_provider() -> None:
    instance = LLMFactory.create(Settings(llm=LLMConfig(provider="azure")))
    assert isinstance(instance, AzureLLM)


def test_factory_routes_deepseek_provider() -> None:
    instance = LLMFactory.create(Settings(llm=LLMConfig(provider="deepseek")))
    assert isinstance(instance, DeepSeekLLM)


def test_factory_passes_deepseek_config_values() -> None:
    instance = LLMFactory.create(
        {
            "llm": {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "api_key": "k",
                "base_url": "https://api.deepseek.com",
            }
        }
    )
    assert isinstance(instance, DeepSeekLLM)
    assert instance.model == "deepseek-v4-flash"
    assert instance.api_key == "k"
    assert instance.base_url == "https://api.deepseek.com"


def test_factory_routes_ollama_provider() -> None:
    instance = LLMFactory.create(Settings(llm=LLMConfig(provider="ollama")))
    assert isinstance(instance, OllamaLLM)


def test_openai_chat_validation_error_contains_provider() -> None:
    llm = OpenAILLM(api_key="k", model="m")
    with pytest.raises(ValueError) as exc_info:
        llm.chat([{"role": "user", "content": ""}])
    assert "openai validation error" in str(exc_info.value)


def test_openai_chat_success_with_mock_request() -> None:
    called: Dict[str, Any] = {}

    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        called["url"] = url
        called["headers"] = headers
        called["payload"] = payload
        called["timeout"] = timeout
        return {"choices": [{"message": {"content": "ok-openai"}}]}

    llm = OpenAILLM(api_key="k", model="m", request_fn=mock_request)
    result = llm.chat([{"role": "user", "content": "ping"}])
    assert result == "ok-openai"
    assert called["url"].endswith("/chat/completions")
    assert called["payload"]["model"] == "m"


def test_azure_chat_success_with_mock_request() -> None:
    called: Dict[str, Any] = {}

    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        called["url"] = url
        called["headers"] = headers
        called["payload"] = payload
        return {"choices": [{"message": {"content": "ok-azure"}}]}

    llm = AzureLLM(
        api_key="azure-key",
        model="my-deployment",
        endpoint="https://example.openai.azure.com",
        api_version="2024-02-15-preview",
        request_fn=mock_request,
    )
    result = llm.chat([{"role": "user", "content": "ping"}])
    assert result == "ok-azure"
    assert "deployments/my-deployment/chat/completions" in called["url"]
    assert "api-version=2024-02-15-preview" in called["url"]
    assert called["headers"]["api-key"] == "azure-key"


def test_deepseek_chat_request_error_contains_provider() -> None:
    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        raise RuntimeError("network down")

    llm = DeepSeekLLM(api_key="k", model="m", request_fn=mock_request)
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat([{"role": "user", "content": "ping"}])
    assert "deepseek request error" in str(exc_info.value)


def test_ollama_chat_success_with_mock_request() -> None:
    called: Dict[str, Any] = {}

    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        called["url"] = url
        called["headers"] = headers
        called["payload"] = payload
        called["timeout"] = timeout
        return {"message": {"content": "ok-ollama"}}

    llm = OllamaLLM(model="llama3.1:8b", base_url="http://localhost:11434", request_fn=mock_request)
    result = llm.chat([{"role": "user", "content": "ping"}])
    assert result == "ok-ollama"
    assert called["url"].endswith("/api/chat")
    assert called["payload"]["model"] == "llama3.1:8b"
    assert called["payload"]["stream"] is False


def test_ollama_chat_request_error_contains_provider() -> None:
    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        raise RuntimeError("service down")

    llm = OllamaLLM(model="m", request_fn=mock_request)
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat([{"role": "user", "content": "ping"}])
    assert "ollama request error" in str(exc_info.value)
