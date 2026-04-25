from typing import Optional

from libs.llm.openai_llm import OpenAICompatibleLLM, RequestFn


class DeepSeekLLM(OpenAICompatibleLLM):
	def __init__(
		self,
		model: str = "deepseek-chat",
		api_key: str = "",
		base_url: str = "https://api.deepseek.com/v1",
		request_fn: Optional[RequestFn] = None,
		timeout: float = 30.0,
	):
		OpenAICompatibleLLM.__init__(
			self,
			provider_name="deepseek",
			model=model,
			api_key=api_key,
			base_url=base_url,
			request_fn=request_fn,
			timeout=timeout,
		)
