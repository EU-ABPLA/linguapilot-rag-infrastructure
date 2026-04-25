import json
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.request import Request, urlopen

from libs.embedding.base_embedding import BaseEmbedding

RequestFn = Callable[[str, Dict[str, str], Dict[str, Any], float], Dict[str, Any]]


class OllamaEmbedding(BaseEmbedding):
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        request_fn: Optional[RequestFn] = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.request_fn = request_fn
        self.timeout = timeout

    def embed(
        self, texts: Sequence[str], trace: Optional[Any] = None
    ) -> List[List[float]]:
        normalized_texts = _validate_texts(texts)
        payload = {"model": self.model, "input": list(normalized_texts)}
        try:
            data = self._request(
                self.base_url + "/api/embed",
                {"Content-Type": "application/json"},
                payload,
                self.timeout,
            )
        except Exception as exc:
            raise RuntimeError("ollama request error: " + str(exc))
        return _extract_embeddings(data)

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


def _validate_texts(texts: Sequence[str]) -> List[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise ValueError("ollama validation error: texts must be a sequence")
    normalized: List[str] = []
    for item in texts:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("ollama validation error: text item must be non-empty")
        normalized.append(item)
    if not normalized:
        raise ValueError("ollama validation error: texts must not be empty")
    return normalized


def _extract_embeddings(data: Dict[str, Any]) -> List[List[float]]:
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        output: List[List[float]] = []
        for vector in embeddings:
            if not isinstance(vector, list) or not vector:
                raise RuntimeError("ollama response error: invalid embeddings")
            row: List[float] = []
            for dim in vector:
                if not isinstance(dim, (int, float)):
                    raise RuntimeError("ollama response error: invalid embedding value")
                row.append(float(dim))
            output.append(row)
        return output

    embedding = data.get("embedding")
    if isinstance(embedding, list) and embedding:
        row: List[float] = []
        for dim in embedding:
            if not isinstance(dim, (int, float)):
                raise RuntimeError("ollama response error: invalid embedding value")
            row.append(float(dim))
        return [row]
    raise RuntimeError("ollama response error: missing embeddings")
