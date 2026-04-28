from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import Fusion
from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.embedding.base_embedding import BaseEmbedding
from libs.vector_store.chroma_store import ChromaStore


class FakeEmbedding(BaseEmbedding):
    def embed(
        self, texts: Sequence[str], trace: Optional[Any] = None
    ) -> List[List[float]]:
        output: List[List[float]] = []
        for text in texts:
            lower = text.lower()
            if "azure" in lower:
                output.append([1.0, 0.0])
            elif "openai" in lower:
                output.append([0.0, 1.0])
            else:
                output.append([0.6, 0.4])
        return output


class BrokenDenseRetriever:
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        raise RuntimeError("dense route error")


class BrokenSparseRetriever:
    def retrieve(
        self,
        keywords: Sequence[str],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        raise RuntimeError("sparse route error")


def _prepare_store_and_index(tmp_path: Path) -> Tuple[ChromaStore, BM25Indexer]:
    chroma_dir = tmp_path / "db" / "chroma"
    bm25_dir = tmp_path / "db" / "bm25"
    store = ChromaStore(persist_directory=str(chroma_dir), collection="hybrid")
    store.upsert(
        [
            {
                "id": "chunk-azure-guide",
                "vector": [1.0, 0.0],
                "content": "azure setup guide with endpoint notes",
                "metadata": {
                    "source_path": "docs/azure_guide.md",
                    "collection": "cloud",
                    "doc_type": "guide",
                },
            },
            {
                "id": "chunk-azure-faq",
                "vector": [0.7, 0.3],
                "content": "azure faq and troubleshooting notes",
                "metadata": {
                    "source_path": "docs/azure_faq.md",
                    "collection": "cloud",
                    "doc_type": "faq",
                },
            },
            {
                "id": "chunk-openai-guide",
                "vector": [0.0, 1.0],
                "content": "openai setup guide with api examples",
                "metadata": {
                    "source_path": "docs/openai_guide.md",
                    "collection": "ai",
                    "doc_type": "guide",
                },
            },
        ]
    )
    bm25 = BM25Indexer(index_dir=str(bm25_dir))
    bm25.build(
        [
            {
                "chunk_id": "chunk-azure-guide",
                "doc_length": 4,
                "term_weights": {"azure": 3, "setup": 1, "endpoint": 1},
            },
            {
                "chunk_id": "chunk-azure-faq",
                "doc_length": 4,
                "term_weights": {"faq": 2, "troubleshooting": 1, "notes": 1},
            },
            {
                "chunk_id": "chunk-openai-guide",
                "doc_length": 4,
                "term_weights": {"openai": 2, "setup": 1, "guide": 1},
            },
        ],
        persist=True,
    )
    return store, bm25


def _build_hybrid(tmp_path: Path) -> HybridSearch:
    store, bm25 = _prepare_store_and_index(tmp_path)
    settings = {"retrieval": {"top_k": 3}}
    return HybridSearch(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=DenseRetriever(
            settings=settings,
            embedding_client=FakeEmbedding(),
            vector_store=store,
        ),
        sparse_retriever=SparseRetriever(
            settings=settings,
            bm25_indexer=BM25Indexer(index_dir=str((tmp_path / "db" / "bm25"))),
            vector_store=store,
        ),
        fusion=Fusion(settings=settings, rrf_k=60),
    )


def test_hybrid_search_returns_top_k_with_text_and_metadata(tmp_path: Path) -> None:
    hybrid = _build_hybrid(tmp_path)
    results = hybrid.search("How to setup Azure", top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "chunk-azure-guide"
    assert results[0].text
    assert "source_path" in results[0].metadata


def test_hybrid_search_supports_metadata_filters(tmp_path: Path) -> None:
    hybrid = _build_hybrid(tmp_path)
    results = hybrid.search(
        "Azure setup guide",
        top_k=3,
        filters={"collection": "cloud", "doc_type": "guide"},
    )
    assert len(results) >= 1
    for item in results:
        assert item.metadata["collection"] == "cloud"
        assert item.metadata["doc_type"] == "guide"


def test_hybrid_search_fallback_when_dense_route_fails(tmp_path: Path) -> None:
    store, _ = _prepare_store_and_index(tmp_path)
    settings = {"retrieval": {"top_k": 3}}
    hybrid = HybridSearch(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=BrokenDenseRetriever(),
        sparse_retriever=SparseRetriever(
            settings=settings,
            bm25_indexer=BM25Indexer(index_dir=str((tmp_path / "db" / "bm25"))),
            vector_store=store,
        ),
        fusion=Fusion(settings=settings, rrf_k=60),
    )
    results = hybrid.search("azure guide", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id.startswith("chunk-azure")


def test_hybrid_search_fallback_when_sparse_route_fails(tmp_path: Path) -> None:
    store, _ = _prepare_store_and_index(tmp_path)
    settings = {"retrieval": {"top_k": 3}}
    hybrid = HybridSearch(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=DenseRetriever(
            settings=settings,
            embedding_client=FakeEmbedding(),
            vector_store=store,
        ),
        sparse_retriever=BrokenSparseRetriever(),
        fusion=Fusion(settings=settings, rrf_k=60),
    )
    results = hybrid.search("openai setup", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "chunk-openai-guide"
