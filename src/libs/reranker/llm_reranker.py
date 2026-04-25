import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from libs.reranker.base_reranker import BaseReranker

LLMCall = Callable[[str], str]


class LLMReranker(BaseReranker):
	def __init__(
		self,
		llm_call: Optional[LLMCall] = None,
		prompt_path: str = "config/prompts/rerank.txt",
		prompt_text: Optional[str] = None,
	):
		self.llm_call = llm_call
		self.prompt_path = prompt_path
		if prompt_text is None:
			self.prompt_text = self._load_prompt(prompt_path)
		else:
			self.prompt_text = prompt_text

	def rerank(
		self,
		query: str,
		candidates: Sequence[Mapping[str, Any]],
		trace: Optional[Any] = None,
	) -> List[Mapping[str, Any]]:
		if not isinstance(query, str) or not query.strip():
			raise ValueError("llm reranker validation error: query must be non-empty")
		normalized = _normalize_candidates(candidates)
		if not normalized:
			return []
		if self.llm_call is None:
			raise RuntimeError("llm reranker fallback required: llm_call is not configured")
		prompt = self._build_prompt(query, normalized)
		raw = self.llm_call(prompt)
		ordered_ids = _parse_ranked_ids(raw)
		candidate_ids = [item["id"] for item in normalized]
		_validate_ranked_ids(ordered_ids, candidate_ids)
		order_map: Dict[str, Mapping[str, Any]] = {item["id"]: item for item in normalized}
		output: List[Mapping[str, Any]] = []
		used = set()
		for candidate_id in ordered_ids:
			output.append(order_map[candidate_id])
			used.add(candidate_id)
		for item in normalized:
			if item["id"] not in used:
				output.append(item)
		return output

	def _load_prompt(self, prompt_path: str) -> str:
		path = Path(prompt_path)
		if not path.exists():
			raise FileNotFoundError("llm reranker prompt not found: " + prompt_path)
		text = path.read_text(encoding="utf-8").strip()
		if not text:
			raise ValueError("llm reranker prompt is empty")
		return text

	def _build_prompt(
		self, query: str, candidates: Sequence[Mapping[str, Any]]
	) -> str:
		payload = {
			"query": query,
			"candidates": [
				{
					"id": str(item["id"]),
					"content": str(item.get("content", "")),
				}
				for item in candidates
			],
		}
		return (
			self.prompt_text
			+ "\n\nReturn strict JSON only: {\"ranked_ids\": [\"id1\", \"id2\"]}\n\n"
			+ json.dumps(payload, ensure_ascii=True)
		)


def _normalize_candidates(
	candidates: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
	normalized: List[Mapping[str, Any]] = []
	for item in candidates:
		if not isinstance(item, Mapping):
			raise ValueError("llm reranker validation error: candidate must be a mapping")
		candidate_id = item.get("id")
		if not isinstance(candidate_id, str) or not candidate_id.strip():
			raise ValueError("llm reranker validation error: candidate id must be non-empty")
		normalized.append(dict(item))
	return normalized


def _parse_ranked_ids(raw: str) -> List[str]:
	try:
		parsed = json.loads(raw)
	except Exception as exc:
		raise ValueError("llm reranker response error: invalid json: " + str(exc))
	if not isinstance(parsed, Mapping):
		raise ValueError("llm reranker response error: json must be an object")
	ranked_ids = parsed.get("ranked_ids")
	if not isinstance(ranked_ids, list):
		raise ValueError("llm reranker response error: ranked_ids must be a list")
	output: List[str] = []
	for item in ranked_ids:
		if not isinstance(item, str) or not item.strip():
			raise ValueError("llm reranker response error: ranked_ids must contain strings")
		output.append(item)
	return output


def _validate_ranked_ids(ranked_ids: Sequence[str], candidate_ids: Sequence[str]) -> None:
	candidate_set = set(candidate_ids)
	seen = set()
	for item in ranked_ids:
		if item not in candidate_set:
			raise ValueError("llm reranker response error: unknown candidate id: " + item)
		if item in seen:
			raise ValueError("llm reranker response error: duplicate candidate id: " + item)
		seen.add(item)
