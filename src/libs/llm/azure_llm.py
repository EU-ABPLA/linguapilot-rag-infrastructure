from typing import Any, Dict, Optional

from libs.llm.openai_llm import OpenAICompatibleLLM, RequestFn


class AzureLLM(OpenAICompatibleLLM):
	def __init__(
		self,
		model: str = "gpt-4o-mini",
		api_key: str = "",
		endpoint: str = "https://example.openai.azure.com",
		api_version: str = "2024-02-15-preview",
		request_fn: Optional[RequestFn] = None,
		timeout: float = 30.0,
	):
		self.api_version = api_version
		OpenAICompatibleLLM.__init__(
			self,
			provider_name="azure",
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
			+ "/chat/completions?api-version="
			+ self.api_version
		)

	def _build_headers(self) -> Dict[str, str]:
		return {"api-key": self.api_key, "Content-Type": "application/json"}

	def chat(self, messages):
		self.model = self.model.strip()
		return OpenAICompatibleLLM.chat(self, messages)
