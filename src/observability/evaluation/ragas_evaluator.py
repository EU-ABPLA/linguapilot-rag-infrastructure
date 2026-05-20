from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from libs.evaluator.base_evaluator import BaseEvaluator

RuntimeLoader = Callable[[], Dict[str, Any]]


class RagasEvaluator(BaseEvaluator):
    def __init__(self, runtime_loader: Optional[RuntimeLoader] = None):
        self._runtime_loader = runtime_loader or _load_ragas_runtime

    def evaluate(
        self,
        query: str,
        retrieved_ids: Sequence[str],
        golden_ids: Sequence[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        normalized_query = _normalize_text(query, "query")
        contexts = _normalize_text_list(retrieved_ids, "retrieved_ids")
        ground_truths = _normalize_text_list(golden_ids, "golden_ids")
        ground_truth = ground_truths[0] if ground_truths else ""
        runtime = self._runtime_loader()
        dataset = runtime["dataset_cls"].from_dict(
            {
                "question": [normalized_query],
                "answer": [ground_truth],
                "ground_truth": [ground_truth],
                "contexts": [contexts],
            }
        )
        result = runtime["evaluate_fn"](
            dataset,
            metrics=[
                runtime["faithfulness"],
                runtime["answer_relevancy"],
                runtime["context_precision"],
            ],
        )
        payload = _coerce_evaluate_result(result)
        return {
            "faithfulness": _extract_metric(payload, "faithfulness"),
            "answer_relevancy": _extract_metric(payload, "answer_relevancy"),
            "context_precision": _extract_metric(payload, "context_precision"),
        }


def _load_ragas_runtime() -> Dict[str, Any]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise ImportError(
            "RagasEvaluator requires optional dependencies: ragas and datasets. "
            + "Install with `pip install ragas datasets`."
        ) from exc
    return {
        "dataset_cls": Dataset,
        "evaluate_fn": evaluate,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
    }


def _coerce_evaluate_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        if hasattr(frame, "to_dict"):
            rows = frame.to_dict(orient="records")
            if isinstance(rows, list) and rows:
                first = rows[0]
                if isinstance(first, Mapping):
                    return dict(first)
    if hasattr(result, "to_dict"):
        raw = result.to_dict()
        if isinstance(raw, Mapping):
            return dict(raw)
    raise ValueError("ragas evaluate returned unsupported result type")


def _extract_metric(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("missing numeric ragas metric: " + key)
    return float(value)


def _normalize_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field_name + " must be non-empty")
    return value.strip()


def _normalize_text_list(value: Any, field_name: str) -> List[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(field_name + " must be a sequence of strings")
    output: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        output.append(item.strip())
    return output
