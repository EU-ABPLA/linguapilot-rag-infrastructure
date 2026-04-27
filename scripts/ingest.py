from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from core.settings import SettingsError, load_settings
from ingestion.pipeline import IngestionPipeline
from observability.logger import get_logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline ingestion pipeline")
    parser.add_argument("--path", required=True, help="Source document path")
    parser.add_argument("--collection", default="default", help="Target collection")
    parser.add_argument("--force", action="store_true", help="Reingest even if unchanged")
    parser.add_argument("--config", default="config/settings.yaml", help="Settings file path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = get_logger("scripts.ingest")
    source_path = Path(args.path)
    if not source_path.exists() or not source_path.is_file():
        logger.error("source file not found: " + str(source_path))
        return 1
    try:
        settings = load_settings(args.config)
    except SettingsError as exc:
        logger.error(str(exc))
        return 1
    pipeline = IngestionPipeline(settings)
    try:
        result = pipeline.run(
            str(source_path),
            collection=str(args.collection),
            force=bool(args.force),
        )
    except Exception as exc:
        logger.error(str(exc))
        return 1
    logger.info(
        "ingestion status=%s chunks=%s vectors=%s images=%s",
        result.status,
        result.chunk_count,
        result.vector_count,
        result.image_count,
    )
    print(
        "status="
        + result.status
        + " chunks="
        + str(result.chunk_count)
        + " vectors="
        + str(result.vector_count)
        + " images="
        + str(result.image_count)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
