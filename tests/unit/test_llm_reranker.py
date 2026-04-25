from typing import Any, Dict, List

import pytest

from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.reranker_factory import RerankerFactory


def test_factory_routes_llm_backend() -> None:
	instance = RerankerFactory.create({"rerank": {"provider": "llm"}})
	assert isinstance(instance, LLMReranker)


def test_llm_reranker_applies_ranked_ids_order() -> None:
	called: Dict[str, Any] = {}

	def mock_llm(prompt: str) -> str:
		called["prompt"] = prompt
		return '{"ranked_ids":["b","a"]}'

	reranker = LLMReranker(llm_call=mock_llm, prompt_text="rerank prompt")
	candidates: List[Dict[str, Any]] = [
		{"id": "a", "content": "first"},
		{"id": "b", "content": "second"},
		{"id": "c", "content": "third"},
	]
	result = reranker.rerank("query", candidates)
	assert [item["id"] for item in result] == ["b", "a", "c"]
	assert "rerank prompt" in called["prompt"]
	assert '"query": "query"' in called["prompt"]


def test_llm_reranker_rejects_invalid_json_response() -> None:
	reranker = LLMReranker(llm_call=lambda prompt: "not-json", prompt_text="p")
	with pytest.raises(ValueError) as exc_info:
		reranker.rerank("query", [{"id": "a", "content": "x"}])
	assert "llm reranker response error: invalid json" in str(exc_info.value)


def test_llm_reranker_rejects_unknown_ranked_id() -> None:
	reranker = LLMReranker(
		llm_call=lambda prompt: '{"ranked_ids":["missing"]}',
		prompt_text="p",
	)
	with pytest.raises(ValueError) as exc_info:
		reranker.rerank("query", [{"id": "a", "content": "x"}])
	assert "llm reranker response error: unknown candidate id" in str(exc_info.value)


def test_llm_reranker_requires_llm_call_for_execution() -> None:
	reranker = LLMReranker(prompt_text="p")
	with pytest.raises(RuntimeError) as exc_info:
		reranker.rerank("query", [{"id": "a", "content": "x"}])
	assert "llm reranker fallback required" in str(exc_info.value)
