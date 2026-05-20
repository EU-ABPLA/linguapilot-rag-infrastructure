from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from core.secrets import safe_error_message
from core.settings import SettingsError, load_settings
from core.trace.trace_context import TraceContext
from ingestion.pipeline import IngestionPipeline
from observability.logger import get_logger, write_trace


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
    if not source_path.exists():
        logger.error("source path not found: " + str(source_path))
        return 1
    source_files = _resolve_source_files(source_path)
    if not source_files:
        logger.error("no source files found under: " + str(source_path))
        return 1
    try:
        settings = load_settings(args.config)
    except SettingsError as exc:
        logger.error(safe_error_message(exc))
        return 1
    pipeline = IngestionPipeline(settings)
    total_chunks = 0
    total_vectors = 0
    total_images = 0
    status_counts = {"success": 0, "skipped": 0, "failed": 0}
    for file_path in source_files:
        trace = TraceContext(
            trace_type="ingestion",
            metadata={
                "source_path": str(file_path),
                "collection": str(args.collection),
                "force": bool(args.force),
            },
        )
        try:
            result = pipeline.run(
                str(file_path),
                collection=str(args.collection),
                force=bool(args.force),
                trace=trace,
            )
        except Exception as exc:
            error_message = safe_error_message(exc)
            status_counts["failed"] += 1
            logger.error(error_message)
            trace.finish(error=error_message)
            if settings.observability.enabled:
                write_trace(trace.to_dict(), log_file=settings.observability.log_file)
            print(
                "file="
                + str(file_path)
                + " status=failed chunks=0 vectors=0 images=0 error="
                + error_message
            )
            continue
        trace.finish(error=None)
        if settings.observability.enabled:
            write_trace(trace.to_dict(), log_file=settings.observability.log_file)
        if result.status == "success":
            status_counts["success"] += 1
        elif result.status == "skipped":
            status_counts["skipped"] += 1
        else:
            status_counts["failed"] += 1
        total_chunks += int(result.chunk_count)
        total_vectors += int(result.vector_count)
        total_images += int(result.image_count)
        logger.info(
            "ingestion file=%s status=%s chunks=%s vectors=%s images=%s",
            str(file_path),
            result.status,
            result.chunk_count,
            result.vector_count,
            result.image_count,
        )
        print(
            "file="
            + str(file_path)
            + " status="
            + result.status
            + " chunks="
            + str(result.chunk_count)
            + " vectors="
            + str(result.vector_count)
            + " images="
            + str(result.image_count)
        )
    overall = "success"
    if status_counts["failed"] > 0 and status_counts["success"] == 0 and status_counts["skipped"] == 0:
        overall = "failed"
    elif status_counts["failed"] > 0:
        overall = "partial"
    elif status_counts["success"] == 0 and status_counts["skipped"] > 0:
        overall = "skipped"
    print(
        "status="
        + overall
        + " files="
        + str(len(source_files))
        + " success="
        + str(status_counts["success"])
        + " skipped="
        + str(status_counts["skipped"])
        + " failed="
        + str(status_counts["failed"])
        + " chunks="
        + str(total_chunks)
        + " vectors="
        + str(total_vectors)
        + " images="
        + str(total_images)
    )
    if overall == "failed":
        return 1
    return 0


def _resolve_source_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    output: List[Path] = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            output.append(item)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
