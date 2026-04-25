from typing import Any, Callable, List, Mapping, Optional, Sequence

from libs.reranker.base_reranker import BaseReranker

ScorerFn = Callable[[str, Mapping[str, Any]], float]


class CrossEncoderReranker(BaseReranker):
	def __init__(
		self,
		scorer: Optional[ScorerFn] = None,
		max_candidates: int = 20,
	):
		self.scorer = scorer
		self.max_candidates = max(1, int(max_candidates))

	def rerank(
		self,
		query: str,
		candidates: Sequence[Mapping[str, Any]],
		trace: Optional[Any] = None,
	) -> List[Mapping[str, Any]]:
		if not isinstance(query, str) or not query.strip():
			raise ValueError("cross_encoder validation error: query must be non-empty")
		normalized = _normalize_candidates(candidates)
		if not normalized:
			return []
		if self.scorer is None:
			raise RuntimeError(
				"cross_encoder fallback required: scorer is not configured"
			)
		head = list(normalized[: self.max_candidates])
		tail = list(normalized[self.max_candidates :])
		try:
			scored = []
			for item in head:
				score = self.scorer(query, item)
				if not isinstance(score, (int, float)):
					raise ValueError(
						"cross_encoder response error: scorer must return numeric score"
					)
				scored.append((float(score), item))
		except TimeoutError as exc:
			raise RuntimeError("cross_encoder fallback required: timeout: " + str(exc))
		except Exception as exc:
			raise RuntimeError("cross_encoder fallback required: " + str(exc))
		scored.sort(key=lambda pair: pair[0], reverse=True)
		output = [item for _, item in scored]
		output.extend(tail)
		return output


def _normalize_candidates(
	candidates: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
	normalized: List[Mapping[str, Any]] = []
	for item in candidates:
		if not isinstance(item, Mapping):
			raise ValueError("cross_encoder validation error: candidate must be a mapping")
		candidate_id = item.get("id")
		if not isinstance(candidate_id, str) or not candidate_id.strip():
			raise ValueError(
				"cross_encoder validation error: candidate id must be non-empty"
			)
		normalized.append(dict(item))
	return normalized
