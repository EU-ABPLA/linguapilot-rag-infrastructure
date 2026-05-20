from __future__ import annotations

from typing import Any, Dict, List

from core.settings import Settings, load_settings
from libs.vector_store.chroma_store import ChromaStore


class ConfigService:
    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        chroma_persist_directory: str = "data/db/chroma",
    ):
        self._settings_path = settings_path
        self._chroma_persist_directory = chroma_persist_directory

    def get_component_cards(self) -> List[Dict[str, Any]]:
        settings = load_settings(self._settings_path)
        return [
            {
                "name": "LLM",
                "provider": settings.llm.provider,
                "model": settings.llm.model,
                "enabled": True,
                "details": {"api_key_configured": bool(settings.llm.api_key)},
            },
            {
                "name": "Embedding",
                "provider": settings.embedding.provider,
                "model": settings.embedding.model,
                "enabled": True,
                "details": {"dimensions": settings.embedding.dimensions},
            },
            {
                "name": "Vector Store",
                "provider": settings.vector_store.provider,
                "model": "",
                "enabled": True,
                "details": {"collection": settings.vector_store.collection},
            },
            {
                "name": "Rerank",
                "provider": settings.rerank.provider,
                "model": settings.rerank.model,
                "enabled": settings.rerank.enabled,
                "details": {},
            },
        ]

    def get_collection_stats(self) -> Dict[str, Any]:
        settings = load_settings(self._settings_path)
        store = ChromaStore(
            persist_directory=self._chroma_persist_directory,
            collection=settings.vector_store.collection,
        )
        stats = store.get_collection_stats()
        return {
            "collection": settings.vector_store.collection,
            "vector_records": int(stats.get("vector_records", 0)),
            "unique_sources": int(stats.get("unique_sources", 0)),
        }
