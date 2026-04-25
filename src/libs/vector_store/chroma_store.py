import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from libs.vector_store.base_vector_store import BaseVectorStore


class ChromaStore(BaseVectorStore):
	def __init__(
		self,
		persist_directory: str = "data/db/chroma",
		collection: str = "default",
	):
		self.persist_directory = Path(persist_directory)
		self.persist_directory.mkdir(parents=True, exist_ok=True)
		self.collection = collection
		self._db_file = self.persist_directory / (self.collection + ".json")
		self._records: Dict[str, Dict[str, Any]] = {}
		self._load()

	def upsert(
		self, records: Sequence[Mapping[str, Any]], trace: Optional[Any] = None
	) -> None:
		for record in records:
			record_id = str(record.get("id", "")).strip()
			if not record_id:
				raise ValueError("chroma validation error: id must be non-empty")
			vector_value = record.get("vector")
			if not isinstance(vector_value, Sequence):
				raise ValueError("chroma validation error: vector must be a sequence")
			vector = _normalize_vector(vector_value)
			content = str(record.get("content", ""))
			metadata_value = record.get("metadata", {})
			if not isinstance(metadata_value, Mapping):
				raise ValueError("chroma validation error: metadata must be a mapping")
			metadata = dict(metadata_value)
			self._records[record_id] = {
				"id": record_id,
				"vector": vector,
				"content": content,
				"metadata": metadata,
			}
		self._save()

	def query(
		self,
		vector: Sequence[float],
		top_k: int,
		filters: Optional[Mapping[str, Any]] = None,
		trace: Optional[Any] = None,
	) -> List[Mapping[str, Any]]:
		if top_k <= 0:
			raise ValueError("top_k must be positive")
		query_vector = _normalize_vector(vector)
		ranked: List[Dict[str, Any]] = []
		for item in self._records.values():
			metadata = item["metadata"]
			if filters is not None and not _matches_filter(metadata, filters):
				continue
			score = _cosine_similarity(query_vector, item["vector"])
			ranked.append(
				{
					"id": item["id"],
					"score": score,
					"content": item["content"],
					"metadata": dict(metadata),
				}
			)
		ranked.sort(key=lambda x: x["score"], reverse=True)
		return ranked[:top_k]

	def _load(self) -> None:
		if not self._db_file.exists():
			return
		raw = self._db_file.read_text(encoding="utf-8")
		parsed = json.loads(raw) if raw.strip() else {}
		if not isinstance(parsed, dict):
			raise RuntimeError("chroma persistence error: invalid data")
		records: Dict[str, Dict[str, Any]] = {}
		for key, item in parsed.items():
			if not isinstance(item, Mapping):
				continue
			vector_value = item.get("vector", [])
			metadata_value = item.get("metadata", {})
			if not isinstance(metadata_value, Mapping):
				metadata_value = {}
			try:
				vector = _normalize_vector(vector_value)
			except ValueError:
				continue
			records[str(key)] = {
				"id": str(item.get("id", key)),
				"vector": vector,
				"content": str(item.get("content", "")),
				"metadata": dict(metadata_value),
			}
		self._records = records

	def _save(self) -> None:
		payload = json.dumps(self._records, ensure_ascii=True, indent=2, sort_keys=True)
		self._db_file.write_text(payload, encoding="utf-8")


def _normalize_vector(vector: Sequence[Any]) -> List[float]:
	values: List[float] = []
	for dim in vector:
		if not isinstance(dim, (int, float)):
			raise ValueError("chroma validation error: vector must contain numeric values")
		values.append(float(dim))
	if not values:
		raise ValueError("chroma validation error: vector must not be empty")
	return values


def _matches_filter(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
	for key, value in filters.items():
		if metadata.get(key) != value:
			return False
	return True


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
	if len(a) != len(b):
		min_size = min(len(a), len(b))
		if min_size == 0:
			return 0.0
		a = a[:min_size]
		b = b[:min_size]
	dot = 0.0
	norm_a = 0.0
	norm_b = 0.0
	for left, right in zip(a, b):
		dot += left * right
		norm_a += left * left
		norm_b += right * right
	if norm_a <= 0.0 or norm_b <= 0.0:
		return 0.0
	return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
