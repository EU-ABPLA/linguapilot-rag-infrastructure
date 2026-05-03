from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

from libs.evaluator.base_evaluator import BaseEvaluator


class CompositeEvaluator(BaseEvaluator):
    def __init__(self, evaluators: List[BaseEvaluator]):
        if not isinstance(evaluators, list) or not evaluators:
            raise ValueError("evaluators must be a non-empty list")
        checked: List[BaseEvaluator] = []
        for item in evaluators:
            if not isinstance(item, BaseEvaluator):
                raise ValueError("evaluators must contain BaseEvaluator")
            checked.append(item)
        self._evaluators = checked

    def evaluate(
        self,
        query: str,
        retrieved_ids: Sequence[str],
        golden_ids: Sequence[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        if len(self._evaluators) == 1:
            return self._evaluators[0].evaluate(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=golden_ids,
                trace=trace,
            )
        output: Dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=len(self._evaluators)) as executor:
            futures = {
                executor.submit(
                    _run_evaluator,
                    evaluator,
                    query,
                    retrieved_ids,
                    golden_ids,
                    trace,
                ): evaluator
                for evaluator in self._evaluators
            }
            for future in as_completed(futures):
                evaluator = futures[future]
                metrics = future.result()
                prefix = _backend_key(evaluator)
                for key, value in metrics.items():
                    if key not in output:
                        output[key] = value
                        continue
                    output[prefix + "." + key] = value
        return output


def _run_evaluator(
    evaluator: BaseEvaluator,
    query: str,
    retrieved_ids: Sequence[str],
    golden_ids: Sequence[str],
    trace: Optional[Any],
) -> Dict[str, float]:
    metrics = evaluator.evaluate(
        query=query,
        retrieved_ids=retrieved_ids,
        golden_ids=golden_ids,
        trace=trace,
    )
    return _normalize_metrics(metrics)


def _normalize_metrics(metrics: Any) -> Dict[str, float]:
    if not isinstance(metrics, dict):
        raise ValueError("evaluator must return dict metrics")
    output: Dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metric name must be non-empty string")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("metric value must be numeric")
        output[key.strip()] = float(value)
    return output


def _backend_key(evaluator: BaseEvaluator) -> str:
    name = evaluator.__class__.__name__.strip().lower()
    if not name:
        return "evaluator"
    return name
