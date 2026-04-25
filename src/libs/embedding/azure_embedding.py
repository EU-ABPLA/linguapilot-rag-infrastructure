from typing import Any, Dict, Optional, Sequence

from libs.embedding.openai_embedding import OpenAIEmbedding, RequestFn


class AzureEmbedding(OpenAIEmbedding):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        endpoint: str = "https://example.openai.azure.com",
        api_version: str = "2024-02-15-preview",
        request_fn: Optional[RequestFn] = None,
        timeout: float = 30.0,
    ):
        self.api_version = api_version
        OpenAIEmbedding.__init__(
            self,
            model=model,
            api_key=api_key,
            base_url=endpoint,
            request_fn=request_fn,
            timeout=timeout,
        )

    def _build_url(self) -> str:
        return (
            self.base_url
            + "/openai/deployments/"
            + self.model
            + "/embeddings?api-version="
            + self.api_version
        )

    def _build_headers(self) -> Dict[str, str]:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    def _build_payload(self, texts: Sequence[str]) -> Dict[str, Any]:
        return {"input": list(texts)}

    def embed(self, texts, trace=None):
        self.model = self.model.strip()
        try:
            return OpenAIEmbedding.embed(self, texts, trace)
        except RuntimeError as exc:
            message = str(exc)
            if message.startswith("openai request error:"):
                raise RuntimeError("azure request error:" + message[len("openai request error:") :])
            if message.startswith("openai response error:"):
                raise RuntimeError("azure response error:" + message[len("openai response error:") :])
            raise
        except ValueError as exc:
            message = str(exc)
            if message.startswith("openai validation error:"):
                raise ValueError("azure validation error:" + message[len("openai validation error:") :])
            raise
