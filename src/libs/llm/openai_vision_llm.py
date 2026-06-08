from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union
from urllib.request import Request, urlopen

from libs.llm.base_vision_llm import BaseVisionLLM

RequestFn = Callable[[str, Dict[str, str], Dict[str, Any], float], Dict[str, Any]]
ImageProcessor = Callable[[bytes, int], bytes]


class OpenAICompatibleVisionLLM(BaseVisionLLM):
    def __init__(
        self,
        provider_name: str,
        model: str,
        api_key: str,
        base_url: str,
        request_fn: Optional[RequestFn] = None,
        image_processor: Optional[ImageProcessor] = None,
        max_image_size: int = 2048,
        timeout: float = 30.0,
    ):
        self.provider_name = provider_name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.request_fn = request_fn
        self.image_processor = image_processor
        self.max_image_size = max(1, int(max_image_size))
        self.timeout = timeout

    def chat_with_image(
        self,
        text: str,
        image_path: Union[str, bytes],
        trace: Optional[Any] = None,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(self.provider_name + " vision validation error: text must be non-empty")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(self.provider_name + " vision validation error: model must be non-empty")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError(self.provider_name + " vision validation error: base_url must be non-empty")
        image_bytes = self._prepare_image(self._read_image_bytes(image_path))
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(image_bytes)},
                        },
                    ],
                }
            ],
        }
        try:
            data = self._request(
                self._build_url(), self._build_headers(), payload, self.timeout
            )
        except Exception as exc:
            raise RuntimeError(self.provider_name + " vision request error: " + str(exc))
        return _extract_content(data, self.provider_name)

    def _build_url(self) -> str:
        return self.base_url + "/chat/completions"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }

    def _request(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        if self.request_fn is not None:
            return self.request_fn(url, headers, payload, timeout)
        body = json.dumps(payload).encode("utf-8")
        request = Request(url=url, data=body, headers=headers, method="POST")
        response = urlopen(request, timeout=timeout)
        parsed = json.loads(response.read().decode("utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError("invalid response type")
        return parsed

    def _read_image_bytes(self, image_path: Union[str, bytes]) -> bytes:
        if isinstance(image_path, bytes):
            if not image_path:
                raise ValueError(self.provider_name + " vision validation error: image bytes must be non-empty")
            return image_path
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(self.provider_name + " vision validation error: image path must be non-empty")
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(self.provider_name + " vision image not found: " + image_path)
        data = path.read_bytes()
        if not data:
            raise ValueError(self.provider_name + " vision validation error: image file is empty")
        return data

    def _prepare_image(self, image_bytes: bytes) -> bytes:
        max_bytes = self.max_image_size * self.max_image_size
        if len(image_bytes) <= max_bytes:
            return image_bytes
        if self.image_processor is None:
            return image_bytes
        processed = self.image_processor(image_bytes, self.max_image_size)
        if not isinstance(processed, bytes) or not processed:
            raise ValueError(self.provider_name + " vision validation error: image processor returned invalid bytes")
        return processed


def _data_url(image_bytes: bytes) -> str:
    return (
        "data:"
        + _detect_mime_type(image_bytes)
        + ";base64,"
        + base64.b64encode(image_bytes).decode("utf-8")
    )


def _detect_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _extract_content(data: Dict[str, Any], provider_name: str) -> str:
    error = data.get("error")
    if isinstance(error, Mapping):
        code = error.get("code")
        message = error.get("message")
        raise RuntimeError(
            provider_name
            + " vision response error: code="
            + str(code)
            + " message="
            + str(message)
        )
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(provider_name + " vision response error: missing choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise RuntimeError(provider_name + " vision response error: invalid choice")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError(provider_name + " vision response error: missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(provider_name + " vision response error: invalid content")
    return content
