from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class BM25Indexer:
    def __init__(
        self,
        index_dir: str = "data/db/bm25",
        index_file: str = "bm25_index.pkl",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / index_file
        self.k1 = float(k1)
        self.b = float(b)
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._index_payload: Dict[str, Any] = _empty_payload()

    def build(
        self, sparse_stats: Sequence[Mapping[str, Any]], persist: bool = True
    ) -> Dict[str, Any]:
        self._documents = {}
        self._merge_documents(sparse_stats)
        self._rebuild_index()
        if persist:
            self.save()
        return self.snapshot()

    def update(
        self, sparse_stats: Sequence[Mapping[str, Any]], persist: bool = True
    ) -> Dict[str, Any]:
        self._merge_documents(sparse_stats)
        self._rebuild_index()
        if persist:
            self.save()
        return self.snapshot()

    def save(self) -> None:
        with self.index_file.open("wb") as handle:
            pickle.dump(
                {
                    "documents": self._documents,
                    "index_payload": self._index_payload,
                    "k1": self.k1,
                    "b": self.b,
                },
                handle,
            )

    def load(self) -> Dict[str, Any]:
        if not self.index_file.exists():
            self._documents = {}
            self._index_payload = _empty_payload()
            return self.snapshot()
        with self.index_file.open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, Mapping):
            raise RuntimeError("bm25 index load error: invalid payload")
        documents = data.get("documents", {})
        index_payload = data.get("index_payload", {})
        self._documents = _normalize_documents_payload(documents)
        self._index_payload = _normalize_index_payload(index_payload)
        if "k1" in data:
            self.k1 = float(data["k1"])
        if "b" in data:
            self.b = float(data["b"])
        return self.snapshot()

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        tokens = [token.lower() for token in _TOKEN_PATTERN.findall(query)]
        if not tokens:
            return []
        query_term_freq: Dict[str, int] = {}
        for token in tokens:
            query_term_freq[token] = query_term_freq.get(token, 0) + 1
        scores: Dict[str, float] = {}
        index = self._index_payload.get("index", {})
        avg_doc_length = float(self._index_payload.get("avg_doc_length", 0.0))
        if avg_doc_length <= 0.0:
            avg_doc_length = 1.0
        for term, qtf in query_term_freq.items():
            term_data = index.get(term)
            if not isinstance(term_data, Mapping):
                continue
            idf = float(term_data.get("idf", 0.0))
            postings = term_data.get("postings", [])
            if not isinstance(postings, list):
                continue
            for posting in postings:
                if not isinstance(posting, Mapping):
                    continue
                chunk_id = posting.get("chunk_id")
                tf = posting.get("tf")
                doc_length = posting.get("doc_length")
                if not isinstance(chunk_id, str) or not chunk_id:
                    continue
                if not isinstance(tf, int) or tf <= 0:
                    continue
                if not isinstance(doc_length, int) or doc_length < 0:
                    continue
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (float(doc_length) / avg_doc_length)
                )
                if denominator <= 0:
                    continue
                increment = idf * ((tf * (self.k1 + 1.0)) / denominator) * float(qtf)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + increment
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            {"chunk_id": chunk_id, "score": float(score)}
            for chunk_id, score in ranked[:top_k]
        ]

    def remove_document(self, chunk_id: str, persist: bool = True) -> bool:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_id must be non-empty")
        normalized = chunk_id.strip()
        if normalized not in self._documents:
            return False
        self._documents.pop(normalized, None)
        self._rebuild_index()
        if persist:
            self.save()
        return True

    def snapshot(self) -> Dict[str, Any]:
        return {
            "doc_count": int(self._index_payload.get("doc_count", 0)),
            "avg_doc_length": float(self._index_payload.get("avg_doc_length", 0.0)),
            "index": _clone_mapping(self._index_payload.get("index", {})),
        }

    def _merge_documents(self, sparse_stats: Sequence[Mapping[str, Any]]) -> None:
        for item in sparse_stats:
            normalized = _normalize_sparse_item(item)
            self._documents[normalized["chunk_id"]] = normalized

    def _rebuild_index(self) -> None:
        doc_count = len(self._documents)
        if doc_count == 0:
            self._index_payload = _empty_payload()
            return
        doc_lengths = {
            chunk_id: int(record["doc_length"])
            for chunk_id, record in self._documents.items()
        }
        avg_doc_length = sum(doc_lengths.values()) / float(doc_count)
        term_postings: Dict[str, List[Dict[str, Any]]] = {}
        for chunk_id, record in self._documents.items():
            term_weights = record["term_weights"]
            for term, tf in term_weights.items():
                term_postings.setdefault(term, []).append(
                    {
                        "chunk_id": chunk_id,
                        "tf": int(tf),
                        "doc_length": int(record["doc_length"]),
                    }
                )
        index: Dict[str, Dict[str, Any]] = {}
        for term, postings in term_postings.items():
            df = len(postings)
            idf = math.log((doc_count - df + 0.5) / (df + 0.5))
            index[term] = {"idf": float(idf), "postings": postings}
        self._index_payload = {
            "doc_count": doc_count,
            "avg_doc_length": float(avg_doc_length),
            "index": index,
        }


def _normalize_sparse_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    chunk_id = item.get("chunk_id")
    doc_length = item.get("doc_length")
    term_weights = item.get("term_weights")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("sparse stats item missing valid chunk_id")
    if not isinstance(doc_length, int) or doc_length < 0:
        raise ValueError("sparse stats item missing valid doc_length")
    if not isinstance(term_weights, Mapping):
        raise ValueError("sparse stats item missing valid term_weights")
    normalized_weights: Dict[str, int] = {}
    for term, value in term_weights.items():
        if not isinstance(term, str) or not term:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        normalized_weights[term.lower()] = int(value)
    return {
        "chunk_id": chunk_id,
        "doc_length": int(doc_length),
        "term_weights": normalized_weights,
    }


def _empty_payload() -> Dict[str, Any]:
    return {"doc_count": 0, "avg_doc_length": 0.0, "index": {}}


def _clone_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    output: Dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            output[str(key)] = _clone_mapping(item)
        elif isinstance(item, list):
            output[str(key)] = _clone_list(item)
        else:
            output[str(key)] = item
    return output


def _clone_list(value: Sequence[Any]) -> List[Any]:
    output: List[Any] = []
    for item in value:
        if isinstance(item, Mapping):
            output.append(_clone_mapping(item))
        elif isinstance(item, list):
            output.append(_clone_list(item))
        else:
            output.append(item)
    return output


def _normalize_documents_payload(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for _, item in value.items():
        if isinstance(item, Mapping):
            normalized = _normalize_sparse_item(item)
            output[normalized["chunk_id"]] = normalized
    return output


def _normalize_index_payload(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return _empty_payload()
    payload = _clone_mapping(value)
    doc_count = payload.get("doc_count", 0)
    avg_doc_length = payload.get("avg_doc_length", 0.0)
    index = payload.get("index", {})
    if not isinstance(doc_count, int) or doc_count < 0:
        doc_count = 0
    if isinstance(avg_doc_length, int):
        avg_doc_length = float(avg_doc_length)
    if not isinstance(avg_doc_length, float) or avg_doc_length < 0:
        avg_doc_length = 0.0
    if not isinstance(index, Mapping):
        index = {}
    return {
        "doc_count": doc_count,
        "avg_doc_length": avg_doc_length,
        "index": _clone_mapping(index),
    }
