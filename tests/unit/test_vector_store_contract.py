from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


@dataclass
class VectorStoreConfig:
    provider: str


@dataclass
class Settings:
    vector_store: VectorStoreConfig


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def upsert(
        self, records: Sequence[Mapping[str, Any]], trace: Optional[Any] = None
    ) -> None:
        normalized: List[Dict[str, Any]] = []
        for record in records:
            normalized.append(
                {
                    "id": str(record.get("id", "")),
                    "vector": list(record.get("vector", [])),
                    "content": str(record.get("content", "")),
                    "metadata": dict(record.get("metadata", {})),
                }
            )
        self._records.extend(normalized)

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[Mapping[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        ranked = []
        for item in self._records:
            if filters:
                matched = True
                for key, value in filters.items():
                    if item["metadata"].get(key) != value:
                        matched = False
                        break
                if not matched:
                    continue
            score = float(len(vector))
            ranked.append(
                {
                    "id": item["id"],
                    "score": score,
                    "content": item["content"],
                    "metadata": item["metadata"],
                }
            )
        return ranked[:top_k]

    def get_by_ids(self, ids: Sequence[str]) -> List[Mapping[str, Any]]:
        lookup = set(str(item) for item in ids)
        return [item for item in self._records if item["id"] in lookup]


def test_factory_routes_registered_provider() -> None:
    provider = "fake"
    VectorStoreFactory.register(provider, FakeVectorStore)
    try:
        settings = Settings(vector_store=VectorStoreConfig(provider=provider))
        instance = VectorStoreFactory.create(settings)
        assert isinstance(instance, FakeVectorStore)
    finally:
        VectorStoreFactory.unregister(provider)


def test_factory_supports_mapping_settings() -> None:
    provider = "mapping-provider"
    VectorStoreFactory.register(provider, FakeVectorStore)
    try:
        settings = {"vector_store": {"provider": provider}}
        instance = VectorStoreFactory.create(settings)
        assert isinstance(instance, FakeVectorStore)
    finally:
        VectorStoreFactory.unregister(provider)


def test_factory_raises_for_unknown_provider() -> None:
    settings = Settings(vector_store=VectorStoreConfig(provider="unknown-provider"))
    with pytest.raises(ValueError) as exc_info:
        VectorStoreFactory.create(settings)
    assert "Unknown vector_store provider: unknown_provider" in str(exc_info.value)


def test_vector_store_contract_query_shape() -> None:
    provider = "contract-provider"
    VectorStoreFactory.register(provider, FakeVectorStore)
    try:
        settings = Settings(vector_store=VectorStoreConfig(provider=provider))
        store = VectorStoreFactory.create(settings)
        store.upsert(
            [
                {
                    "id": "chunk-1",
                    "vector": [0.1, 0.2],
                    "content": "hello",
                    "metadata": {"collection": "default"},
                }
            ]
        )
        results = store.query([1.0, 2.0], top_k=1, filters={"collection": "default"})
        assert isinstance(results, list)
        assert len(results) == 1
        result = results[0]
        assert set(["id", "score", "content", "metadata"]).issubset(result.keys())
        assert isinstance(result["id"], str)
        assert isinstance(result["score"], float)
        assert isinstance(result["content"], str)
        assert isinstance(result["metadata"], dict)
    finally:
        VectorStoreFactory.unregister(provider)


def test_vector_store_contract_rejects_invalid_top_k() -> None:
    provider = "invalid-top-k"
    VectorStoreFactory.register(provider, FakeVectorStore)
    try:
        settings = Settings(vector_store=VectorStoreConfig(provider=provider))
        store = VectorStoreFactory.create(settings)
        with pytest.raises(ValueError) as exc_info:
            store.query([1.0], top_k=0)
        assert "top_k must be positive" in str(exc_info.value)
    finally:
        VectorStoreFactory.unregister(provider)


def test_vector_store_contract_get_by_ids_shape() -> None:
    provider = "get-by-ids-provider"
    VectorStoreFactory.register(provider, FakeVectorStore)
    try:
        settings = Settings(vector_store=VectorStoreConfig(provider=provider))
        store = VectorStoreFactory.create(settings)
        store.upsert(
            [
                {
                    "id": "chunk-1",
                    "vector": [0.1, 0.2],
                    "content": "hello",
                    "metadata": {"collection": "default"},
                },
                {
                    "id": "chunk-2",
                    "vector": [0.2, 0.1],
                    "content": "world",
                    "metadata": {"collection": "default"},
                },
            ]
        )
        rows = store.get_by_ids(["chunk-2", "missing", "chunk-1"])
        assert [item["id"] for item in rows] == ["chunk-1", "chunk-2"]
        assert set(["id", "content", "metadata"]).issubset(rows[0].keys())
    finally:
        VectorStoreFactory.unregister(provider)


def test_chroma_delete_by_metadata_rejects_empty_filters(tmp_path: Path) -> None:
    store = ChromaStore(
        persist_directory=str(tmp_path / "db"),
        collection="delete-boundary",
    )
    with pytest.raises(ValueError) as exc_info:
        store.delete_by_metadata({})
    assert "filters must be a non-empty mapping" in str(exc_info.value)


def test_chroma_delete_by_metadata_returns_zero_when_no_match(tmp_path: Path) -> None:
    store = ChromaStore(
        persist_directory=str(tmp_path / "db"),
        collection="delete-no-match",
    )
    store.upsert(
        [
            {
                "id": "chunk-1",
                "vector": [0.1, 0.2],
                "content": "hello",
                "metadata": {"collection": "default", "source_path": "docs/a.md"},
            }
        ]
    )
    deleted = store.delete_by_metadata({"collection": "missing"})
    assert deleted == 0
    rows = store.get_by_metadata()
    assert len(rows) == 1
    assert rows[0]["id"] == "chunk-1"


def test_chroma_delete_by_metadata_removes_matching_records(tmp_path: Path) -> None:
    store = ChromaStore(
        persist_directory=str(tmp_path / "db"),
        collection="delete-hit",
    )
    store.upsert(
        [
            {
                "id": "chunk-1",
                "vector": [0.1, 0.2],
                "content": "hello",
                "metadata": {"collection": "default", "source_path": "docs/a.md"},
            },
            {
                "id": "chunk-2",
                "vector": [0.2, 0.1],
                "content": "world",
                "metadata": {"collection": "test", "source_path": "docs/b.md"},
            },
        ]
    )
    deleted = store.delete_by_metadata({"collection": "default"})
    assert deleted == 1
    rows = store.get_by_metadata()
    assert [item["id"] for item in rows] == ["chunk-2"]
