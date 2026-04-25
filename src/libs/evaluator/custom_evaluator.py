from typing import Any, Dict, Optional, Sequence, Set

from libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
	def evaluate(
		self,
		query: str,
		retrieved_ids: Sequence[str],
		golden_ids: Sequence[str],
		trace: Optional[Any] = None,
	) -> Dict[str, float]:
		golden_set: Set[str] = set(golden_ids)
		if not retrieved_ids:
			return {"hit_rate": 0.0, "mrr": 0.0}

		hit_rate = 0.0
		reciprocal_rank = 0.0
		for index, item_id in enumerate(retrieved_ids):
			if item_id in golden_set:
				hit_rate = 1.0
				reciprocal_rank = 1.0 / float(index + 1)
				break
		return {"hit_rate": hit_rate, "mrr": reciprocal_rank}
