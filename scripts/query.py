from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import Fusion
from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.reranker import Reranker
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import SettingsError, load_settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer
from observability.logger import get_logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run online query pipeline")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--collection", default="", help="Collection filter")
    parser.add_argument("--verbose", action="store_true", help="Show stage-level outputs")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranker stage")
    parser.add_argument("--config", default="config/settings.yaml", help="Settings file path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = get_logger("scripts.query")
    if args.top_k <= 0:
        logger.error("top_k must be positive")
        return 1
    try:
        settings = load_settings(args.config)
    except SettingsError as exc:
        logger.error(str(exc))
        return 1
    query_processor = QueryProcessor()
    dense_retriever = DenseRetriever(settings=settings)
    sparse_retriever = SparseRetriever(settings=settings, bm25_indexer=BM25Indexer())
    fusion = Fusion(settings=settings)
    hybrid_search = HybridSearch(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
    )
    reranker = Reranker(settings=settings)
    filters: Dict[str, Any] = {}
    if isinstance(args.collection, str) and args.collection.strip():
        filters["collection"] = args.collection.strip()
    trace = TraceContext(trace_type="query")
    try:
        if args.verbose:
            output = _run_verbose(
                query=args.query,
                top_k=args.top_k,
                filters=filters,
                query_processor=query_processor,
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                fusion=fusion,
                hybrid_search=hybrid_search,
                reranker=reranker,
                no_rerank=bool(args.no_rerank),
                trace=trace,
                logger=logger,
            )
        else:
            output = _run_default(
                query=args.query,
                top_k=args.top_k,
                filters=filters,
                hybrid_search=hybrid_search,
                reranker=reranker,
                no_rerank=bool(args.no_rerank),
                trace=trace,
            )
    except Exception as exc:
        logger.error(str(exc))
        return 1
    if not output:
        print("未找到相关文档，请先运行 ingest.py 摄取数据")
        return 0
    print(_format_results(output))
    return 0


def _run_default(
    query: str,
    top_k: int,
    filters: Dict[str, Any],
    hybrid_search: HybridSearch,
    reranker: Reranker,
    no_rerank: bool,
    trace: TraceContext,
) -> List[RetrievalResult]:
    hybrid_output = hybrid_search.search(query, top_k=top_k, filters=filters, trace=trace)
    if no_rerank:
        return hybrid_output
    return reranker.rerank(query, hybrid_output, top_k=top_k, trace=trace)


def _run_verbose(
    query: str,
    top_k: int,
    filters: Dict[str, Any],
    query_processor: QueryProcessor,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
    fusion: Fusion,
    hybrid_search: HybridSearch,
    reranker: Reranker,
    no_rerank: bool,
    trace: TraceContext,
    logger: Any,
) -> List[RetrievalResult]:
    processed = query_processor.process(query, filters=filters, trace=trace)
    dense_output: List[RetrievalResult]
    sparse_output: List[RetrievalResult]
    try:
        dense_output = dense_retriever.retrieve(
            processed.query,
            top_k=top_k,
            filters=processed.filters,
            trace=trace,
        )
    except Exception as exc:
        logger.warning("dense route fallback: %s", str(exc))
        dense_output = []
    try:
        sparse_output = sparse_retriever.retrieve(
            processed.keywords,
            top_k=top_k,
            trace=trace,
        )
    except Exception as exc:
        logger.warning("sparse route fallback: %s", str(exc))
        sparse_output = []
    fused_output = fusion.fuse(dense_output, sparse_output, top_k=top_k, trace=trace)
    hybrid_output = hybrid_search._apply_metadata_filters(fused_output, processed.filters)[:top_k]
    if no_rerank:
        rerank_output = hybrid_output
    else:
        rerank_output = reranker.rerank(query, hybrid_output, top_k=top_k, trace=trace)
    print("=== Dense ===")
    print(_format_results(dense_output))
    print("")
    print("=== Sparse ===")
    print(_format_results(sparse_output))
    print("")
    print("=== Fusion ===")
    print(_format_results(hybrid_output))
    print("")
    print("=== Rerank ===")
    print(_format_results(rerank_output))
    print("")
    return rerank_output


def _format_results(results: Sequence[RetrievalResult]) -> str:
    if not results:
        return "(empty)"
    lines: List[str] = []
    for index, item in enumerate(results, start=1):
        source = str(item.metadata.get("source_path", "unknown"))
        page = item.metadata.get("page")
        page_text = ""
        if isinstance(page, int):
            page_text = " page=" + str(page)
        text_preview = item.text.strip().replace("\n", " ")
        if len(text_preview) > 120:
            text_preview = text_preview[:117] + "..."
        lines.append(
            str(index)
            + ". score="
            + format(float(item.score), ".4f")
            + " source="
            + source
            + page_text
            + " text="
            + text_preview
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
