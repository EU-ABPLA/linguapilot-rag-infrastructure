from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.reranker import Reranker
from core.secrets import safe_error_message
from core.settings import SettingsError, load_settings
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.eval_runner import EvalRunner
from observability.logger import get_logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation on golden test set")
    parser.add_argument(
        "--test-set",
        default="",
        help="Path to golden test set json, default from settings",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Settings file path",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = get_logger("scripts.evaluate")
    try:
        settings = load_settings(args.config)
    except SettingsError as exc:
        logger.error(safe_error_message(exc))
        return 1
    test_set_path = str(args.test_set).strip() or settings.evaluation.golden_test_set
    try:
        hybrid_search = HybridSearch(settings=settings)
        reranker = Reranker(settings=settings)
        evaluator = EvaluatorFactory.create(settings)
        runner = EvalRunner(
            settings=settings,
            hybrid_search=hybrid_search,
            evaluator=evaluator,
            reranker=reranker,
        )
        report = runner.run(test_set_path)
    except Exception as exc:
        logger.error(safe_error_message(exc))
        return 1
    payload = report.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info(
        "evaluation done total=%s completed=%s failed=%s hit_rate=%.4f mrr=%.4f",
        payload["total_cases"],
        payload["completed_cases"],
        payload["failed_cases"],
        payload["hit_rate"],
        payload["mrr"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
