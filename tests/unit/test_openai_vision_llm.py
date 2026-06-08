from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict

import pytest

from libs.llm.openai_vision_llm import OpenAICompatibleVisionLLM


def test_openai_vision_request_uses_image_url_payload(tmp_path: Path) -> None:
    called: Dict[str, Any] = {}
    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"\x89PNG\r\n\x1a\nfake-png")

    def mock_request(
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        called["url"] = url
        called["headers"] = headers
        called["payload"] = payload
        return {"choices": [{"message": {"content": "image summary"}}]}

    llm = OpenAICompatibleVisionLLM(
        provider_name="openai",
        model="gpt-4o-mini",
        api_key="k",
        base_url="https://api.openai.com/v1",
        request_fn=mock_request,
    )
    result = llm.chat_with_image("describe", str(image_file))

    assert result == "image summary"
    assert called["url"] == "https://api.openai.com/v1/chat/completions"
    assert called["headers"]["Authorization"] == "Bearer k"
    assert called["payload"]["model"] == "gpt-4o-mini"
    content = called["payload"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    image_url = content[1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]).startswith(b"\x89PNG")


def test_openai_compatible_vision_response_error_is_clear() -> None:
    def mock_request(
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        return {"error": {"code": "invalid_request", "message": "image_url not supported"}}

    llm = OpenAICompatibleVisionLLM(
        provider_name="openai",
        model="gpt-4o-mini",
        api_key="k",
        base_url="https://api.openai.com/v1",
        request_fn=mock_request,
    )
    with pytest.raises(RuntimeError) as exc_info:
        llm.chat_with_image("describe", b"img")
    assert "openai vision response error: code=invalid_request" in str(exc_info.value)
