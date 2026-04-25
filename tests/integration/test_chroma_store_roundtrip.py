from pathlib import Path

from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


def test_factory_routes_chroma_provider() -> None:
	instance = VectorStoreFactory.create({"vector_store": {"provider": "chroma"}})
	assert isinstance(instance, ChromaStore)


def test_chroma_roundtrip_upsert_and_query_with_filters(tmp_path: Path) -> None:
	store = ChromaStore(persist_directory=str(tmp_path), collection="roundtrip")
	store.upsert(
		[
			{
				"id": "doc-1",
				"vector": [1.0, 0.0],
				"content": "alpha",
				"metadata": {"collection": "default", "topic": "a"},
			},
			{
				"id": "doc-2",
				"vector": [0.0, 1.0],
				"content": "beta",
				"metadata": {"collection": "default", "topic": "b"},
			},
			{
				"id": "doc-3",
				"vector": [0.8, 0.2],
				"content": "gamma",
				"metadata": {"collection": "extra", "topic": "a"},
			},
		]
	)

	results = store.query(
		[1.0, 0.0],
		top_k=2,
		filters={"collection": "default"},
	)
	assert len(results) == 2
	assert results[0]["id"] == "doc-1"
	assert results[0]["score"] >= results[1]["score"]
	assert results[0]["metadata"]["collection"] == "default"


def test_chroma_roundtrip_persists_to_disk(tmp_path: Path) -> None:
	store = ChromaStore(persist_directory=str(tmp_path), collection="persisted")
	store.upsert(
		[
			{
				"id": "persist-1",
				"vector": [0.2, 0.8],
				"content": "persist me",
				"metadata": {"collection": "default"},
			}
		]
	)

	reloaded = ChromaStore(persist_directory=str(tmp_path), collection="persisted")
	results = reloaded.query([0.2, 0.8], top_k=1)
	assert len(results) == 1
	assert results[0]["id"] == "persist-1"
	assert results[0]["content"] == "persist me"


def test_chroma_query_rejects_invalid_top_k(tmp_path: Path) -> None:
	store = ChromaStore(persist_directory=str(tmp_path), collection="invalid")
	store.upsert(
		[
			{
				"id": "x",
				"vector": [1.0],
				"content": "x",
				"metadata": {},
			}
		]
	)
	try:
		store.query([1.0], top_k=0)
		assert False
	except ValueError as exc:
		assert "top_k must be positive" in str(exc)
