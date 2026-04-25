import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from urllib.request import Request, urlopen

from libs.embedding.base_embedding import BaseEmbedding

RequestFn = Callable[[str, Dict[str, str], Dict[str, Any], float], Dict[str, Any]]


class OpenAIEmbedding(BaseEmbedding):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        request_fn: Optional[RequestFn] = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.request_fn = request_fn
        self.timeout = timeout

    def embed(
        self, texts: Sequence[str], trace: Optional[Any] = None
    ) -> List[List[float]]:
        normalized_texts = _validate_texts(texts, "openai")
        payload = self._build_payload(normalized_texts)
        try:
            data = self._request(
                self._build_url(), self._build_headers(), payload, self.timeout
            )
        except Exception as exc:
            raise RuntimeError("openai request error: " + str(exc))
        return _extract_embeddings(data, "openai")

    def _build_url(self) -> str:
        return self.base_url + "/embeddings"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }

    def _build_payload(self, texts: Sequence[str]) -> Dict[str, Any]:
        return {"model": self.model, "input": list(texts)}

    def _request(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        if self.request_fn is not None:
            return self.request_fn(url, headers, payload, timeout)
        return _default_request(url, headers, payload, timeout)


def _default_request(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=body, headers=headers, method="POST")
    response = urlopen(request, timeout=timeout)
    data = response.read().decode("utf-8")
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise RuntimeError("invalid response type")
    return parsed


def _validate_texts(texts: Sequence[str], provider_name: str) -> List[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise ValueError(provider_name + " validation error: texts must be a sequence")
    normalized: List[str] = []
    for item in texts:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                provider_name + " validation error: text item must be non-empty"
            )
        normalized.append(item)
    if not normalized:
        raise ValueError(provider_name + " validation error: texts must not be empty")
    return normalized


def _extract_embeddings(data: Dict[str, Any], provider_name: str) -> List[List[float]]:
    entries = data.get("data")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(provider_name + " response error: missing data")
    output: List[List[float]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeError(provider_name + " response error: invalid data item")
        vector = entry.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError(provider_name + " response error: missing embedding")
        row: List[float] = []
        for dim in vector:
            if not isinstance(dim, (int, float)):
                raise RuntimeError(
                    provider_name + " response error: invalid embedding value"
                )
            row.append(float(dim))
        output.append(row)
    return output
