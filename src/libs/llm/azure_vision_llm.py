import base64
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union
from urllib.request import Request, urlopen

from libs.llm.base_vision_llm import BaseVisionLLM

RequestFn = Callable[[str, Dict[str, str], Dict[str, Any], float], Dict[str, Any]]
ImageProcessor = Callable[[bytes, int], bytes]


class AzureVisionLLM(BaseVisionLLM):
    def __init__(
        self,
        deployment_name: str = "gpt-4o",
        api_key: str = "",
        endpoint: str = "https://example.openai.azure.com",
        api_version: str = "2024-02-15-preview",
        max_image_size: int = 2048,
        request_fn: Optional[RequestFn] = None,
        image_processor: Optional[ImageProcessor] = None,
        timeout: float = 30.0,
    ):
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.api_version = api_version
        self.max_image_size = max(1, int(max_image_size))
        self.request_fn = request_fn
        self.image_processor = image_processor
        self.timeout = timeout

    def chat_with_image(
        self,
        text: str,
        image_path: Union[str, bytes],
        trace: Optional[Any] = None,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("azure vision validation error: text must be non-empty")
        image_bytes = self._read_image_bytes(image_path)
        prepared = self._prepare_image(image_bytes)
        image_base64 = base64.b64encode(prepared).decode("utf-8")
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + image_base64},
                        },
                    ],
                }
            ]
        }
        try:
            data = self._request(
                self._build_url(), self._build_headers(), payload, self.timeout
            )
        except TimeoutError as exc:
            raise RuntimeError("azure vision request error: timeout: " + str(exc))
        except Exception as exc:
            raise RuntimeError("azure vision request error: " + str(exc))
        return _extract_content(data)

    def _build_url(self) -> str:
        return (
            self.endpoint
            + "/openai/deployments/"
            + self.deployment_name
            + "/chat/completions?api-version="
            + self.api_version
        )

    def _build_headers(self) -> Dict[str, str]:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    def _read_image_bytes(self, image_path: Union[str, bytes]) -> bytes:
        if isinstance(image_path, bytes):
            if not image_path:
                raise ValueError("azure vision validation error: image bytes must be non-empty")
            return image_path
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("azure vision validation error: image path must be non-empty")
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError("azure vision image not found: " + image_path)
        data = path.read_bytes()
        if not data:
            raise ValueError("azure vision validation error: image file is empty")
        return data

    def _prepare_image(self, image_bytes: bytes) -> bytes:
        max_bytes = self.max_image_size * self.max_image_size
        if len(image_bytes) <= max_bytes:
            return image_bytes
        if self.image_processor is None:
            return image_bytes
        processed = self.image_processor(image_bytes, self.max_image_size)
        if not isinstance(processed, bytes) or not processed:
            raise ValueError("azure vision validation error: image processor returned invalid bytes")
        return processed

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
        data = response.read().decode("utf-8")
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise RuntimeError("invalid response type")
        return parsed


def _extract_content(data: Dict[str, Any]) -> str:
    error = data.get("error")
    if isinstance(error, Mapping):
        code = error.get("code")
        message = error.get("message")
        raise RuntimeError(
            "azure vision response error: code="
            + str(code)
            + " message="
            + str(message)
        )
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("azure vision response error: missing choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise RuntimeError("azure vision response error: invalid choice")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("azure vision response error: missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("azure vision response error: invalid content")
    return content
