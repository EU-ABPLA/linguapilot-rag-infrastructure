import json
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from urllib.request import Request, urlopen

from libs.llm.base_llm import BaseLLM
from libs.llm.openai_llm import _validate_messages

RequestFn = Callable[[str, Dict[str, str], Dict[str, Any], float], Dict[str, Any]]


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        request_fn: Optional[RequestFn] = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.request_fn = request_fn
        self.timeout = timeout

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        normalized_messages = _validate_messages(messages, "ollama")
        payload = {
            "model": self.model,
            "messages": list(normalized_messages),
            "stream": False,
        }
        try:
            data = self._request(
                self.base_url + "/api/chat",
                {"Content-Type": "application/json"},
                payload,
                self.timeout,
            )
        except Exception as exc:
            raise RuntimeError("ollama request error: " + str(exc))
        message = data.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("ollama response error: missing message")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("ollama response error: invalid content")
        return content

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
