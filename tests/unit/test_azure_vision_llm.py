import base64
from pathlib import Path
from typing import Any, Dict

import pytest

from libs.llm.azure_vision_llm import AzureVisionLLM
from libs.llm.llm_factory import LLMFactory


def test_factory_routes_azure_vision_provider() -> None:
    instance = LLMFactory.create_vision_llm({"vision_llm": {"provider": "azure"}})
    assert isinstance(instance, AzureVisionLLM)


def test_chat_with_image_path_success_with_mock_request(tmp_path: Path) -> None:
    called: Dict[str, Any] = {}
    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"fake-png-bytes")

    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        called["url"] = url
        called["headers"] = headers
        called["payload"] = payload
        return {"choices": [{"message": {"content": "ok-vision"}}]}

    llm = AzureVisionLLM(
        deployment_name="gpt-4o",
        api_key="k",
        endpoint="https://example.openai.azure.com",
        request_fn=mock_request,
    )
    result = llm.chat_with_image("describe", str(image_file))
    assert result == "ok-vision"
    assert "deployments/gpt-4o/chat/completions" in called["url"]
    assert called["headers"]["api-key"] == "k"
    image_url = called["payload"]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(image_url.split(",", 1)[1])
    assert decoded == b"fake-png-bytes"


def test_chat_with_image_bytes_triggers_processor_for_large_input() -> None:
    called: Dict[str, Any] = {}
    processor_called: Dict[str, Any] = {}

    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        called["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    def image_processor(raw: bytes, max_image_size: int) -> bytes:
        processor_called["raw"] = raw
        processor_called["max_image_size"] = max_image_size
        return b"compressed"

    llm = AzureVisionLLM(
        max_image_size=2,
        request_fn=mock_request,
        image_processor=image_processor,
    )
    result = llm.chat_with_image("q", b"0123456789")
    assert result == "ok"
    assert processor_called["max_image_size"] == 2
    image_url = called["payload"]["messages"][0]["content"][1]["image_url"]["url"]
    decoded = base64.b64decode(image_url.split(",", 1)[1])
    assert decoded == b"compressed"


def test_chat_with_image_timeout_contains_provider_error() -> None:
    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        raise TimeoutError("slow network")

    llm = AzureVisionLLM(request_fn=mock_request)
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat_with_image("q", b"img")
    assert "azure vision request error: timeout" in str(exc_info.value)


def test_chat_with_image_auth_failure_contains_error_code() -> None:
    def mock_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        return {"error": {"code": "401", "message": "Unauthorized"}}

    llm = AzureVisionLLM(request_fn=mock_request)
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat_with_image("q", b"img")
    message = str(exc_info.value)
    assert "azure vision response error: code=401" in message
