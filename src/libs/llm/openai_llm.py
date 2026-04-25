import json
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from urllib.request import Request, urlopen

from libs.llm.base_llm import BaseLLM

RequestFn = Callable[[str, Dict[str, str], Dict[str, Any], float], Dict[str, Any]]


class OpenAICompatibleLLM(BaseLLM):
	def __init__(
		self,
		provider_name: str,
		model: str,
		api_key: str,
		base_url: str,
		request_fn: Optional[RequestFn] = None,
		timeout: float = 30.0,
	):
		self.provider_name = provider_name
		self.model = model
		self.api_key = api_key
		self.base_url = base_url.rstrip("/")
		self.request_fn = request_fn
		self.timeout = timeout

	def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
		normalized_messages = _validate_messages(messages, self.provider_name)
		payload = {"model": self.model, "messages": normalized_messages}
		try:
			data = self._request(
				self._build_url(), self._build_headers(), payload, self.timeout
			)
		except Exception as exc:
			raise RuntimeError(
				self.provider_name + " request error: " + str(exc)
			)
		return _extract_text(data, self.provider_name)

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
		return _default_request(url, headers, payload, timeout)


class OpenAILLM(OpenAICompatibleLLM):
	def __init__(
		self,
		model: str = "gpt-4o-mini",
		api_key: str = "",
		base_url: str = "https://api.openai.com/v1",
		request_fn: Optional[RequestFn] = None,
		timeout: float = 30.0,
	):
		OpenAICompatibleLLM.__init__(
			self,
			provider_name="openai",
			model=model,
			api_key=api_key,
			base_url=base_url,
			request_fn=request_fn,
			timeout=timeout,
		)


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


def _validate_messages(
	messages: Sequence[Mapping[str, str]], provider_name: str
) -> Sequence[Mapping[str, str]]:
	if not isinstance(messages, Sequence):
		raise ValueError(provider_name + " validation error: messages must be a sequence")
	normalized = []
	for item in messages:
		if not isinstance(item, Mapping):
			raise ValueError(
				provider_name + " validation error: message item must be a mapping"
			)
		role = item.get("role")
		content = item.get("content")
		if not isinstance(role, str) or not role.strip():
			raise ValueError(provider_name + " validation error: role must be non-empty")
		if not isinstance(content, str) or not content.strip():
			raise ValueError(
				provider_name + " validation error: content must be non-empty"
			)
		normalized.append({"role": role, "content": content})
	if not normalized:
		raise ValueError(provider_name + " validation error: messages must not be empty")
	return normalized


def _extract_text(data: Dict[str, Any], provider_name: str) -> str:
	choices = data.get("choices")
	if not isinstance(choices, list) or not choices:
		raise RuntimeError(provider_name + " response error: missing choices")
	first = choices[0]
	if not isinstance(first, Mapping):
		raise RuntimeError(provider_name + " response error: invalid choice")
	message = first.get("message")
	if not isinstance(message, Mapping):
		raise RuntimeError(provider_name + " response error: missing message")
	content = message.get("content")
	if not isinstance(content, str):
		raise RuntimeError(provider_name + " response error: invalid content")
	return content
