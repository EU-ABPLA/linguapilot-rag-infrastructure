from typing import Any, Dict, List

import pytest

from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.reranker_factory import RerankerFactory


def test_factory_routes_cross_encoder_backend() -> None:
	instance = RerankerFactory.create({"rerank": {"provider": "cross_encoder"}})
	assert isinstance(instance, CrossEncoderReranker)


def test_cross_encoder_reranker_orders_by_mock_score() -> None:
	def scorer(query: str, candidate: Dict[str, Any]) -> float:
		return float(candidate["weight"])

	reranker = CrossEncoderReranker(scorer=scorer)
	candidates: List[Dict[str, Any]] = [
		{"id": "a", "content": "first", "weight": 0.2},
		{"id": "b", "content": "second", "weight": 0.9},
		{"id": "c", "content": "third", "weight": 0.5},
	]
	result = reranker.rerank("q", candidates)
	assert [item["id"] for item in result] == ["b", "c", "a"]


def test_cross_encoder_respects_top_m_limit() -> None:
	def scorer(query: str, candidate: Dict[str, Any]) -> float:
		return float(candidate["weight"])

	reranker = CrossEncoderReranker(scorer=scorer, max_candidates=2)
	candidates: List[Dict[str, Any]] = [
		{"id": "a", "content": "first", "weight": 0.2},
		{"id": "b", "content": "second", "weight": 0.9},
		{"id": "c", "content": "third", "weight": 0.5},
	]
	result = reranker.rerank("q", candidates)
	assert [item["id"] for item in result] == ["b", "a", "c"]


def test_cross_encoder_requires_scorer_for_execution() -> None:
	reranker = CrossEncoderReranker()
	with pytest.raises(RuntimeError) as exc_info:
		reranker.rerank("q", [{"id": "a", "content": "x"}])
	assert "cross_encoder fallback required" in str(exc_info.value)


def test_cross_encoder_reports_timeout_as_fallback_signal() -> None:
	def scorer(query: str, candidate: Dict[str, Any]) -> float:
		raise TimeoutError("slow model")

	reranker = CrossEncoderReranker(scorer=scorer)
	with pytest.raises(RuntimeError) as exc_info:
		reranker.rerank("q", [{"id": "a", "content": "x"}])
	assert "cross_encoder fallback required: timeout" in str(exc_info.value)
