import json

import pytest

from core import Chunk, ChunkRecord, Document


def _metadata() -> dict:
    return {
        "source_path": "data/docs/sample.pdf",
        "images": [
            {
                "id": "dochash_1_1",
                "path": "data/images/default/dochash_1_1.png",
                "page": 1,
                "text_offset": 10,
                "text_length": 22,
                "position": {"x": 1, "y": 2},
            }
        ],
    }


def test_document_roundtrip_dict_and_json() -> None:
    doc = Document(id="doc-1", text="hello", metadata=_metadata())
    payload = doc.to_dict()
    restored = Document.from_dict(payload)
    assert restored == doc
    raw = doc.to_json()
    assert isinstance(json.loads(raw), dict)
    assert Document.from_json(raw) == doc


def test_chunk_roundtrip_dict_and_json() -> None:
    chunk = Chunk(
        id="chunk-1",
        text="chunk text",
        metadata=_metadata(),
        start_offset=3,
        end_offset=15,
        source_ref="doc-1",
    )
    assert Chunk.from_dict(chunk.to_dict()) == chunk
    assert Chunk.from_json(chunk.to_json()) == chunk


def test_chunk_record_roundtrip_dict_and_json() -> None:
    record = ChunkRecord(
        id="record-1",
        text="record text",
        metadata=_metadata(),
        dense_vector=[0.1, 0.2],
        sparse_vector=[1, 0, 3],
    )
    assert ChunkRecord.from_dict(record.to_dict()) == record
    assert ChunkRecord.from_json(record.to_json()) == record


def test_types_require_metadata_source_path() -> None:
    with pytest.raises(ValueError) as doc_error:
        Document(id="d", text="t", metadata={})
    assert "metadata.source_path" in str(doc_error.value)

    with pytest.raises(ValueError) as chunk_error:
        Chunk(id="c", text="t", metadata={})
    assert "metadata.source_path" in str(chunk_error.value)


def test_metadata_images_shape_validation() -> None:
    bad_metadata = {
        "source_path": "x",
        "images": [{"id": "img-1"}],
    }
    with pytest.raises(ValueError) as exc_info:
        Document(id="doc", text="text", metadata=bad_metadata)
    assert "metadata.images[].path" in str(exc_info.value)


def test_chunk_offsets_are_validated() -> None:
    with pytest.raises(ValueError) as exc_info:
        Chunk(
            id="chunk-1",
            text="x",
            metadata=_metadata(),
            start_offset=9,
            end_offset=3,
        )
    assert "end_offset must be greater than or equal to start_offset" in str(exc_info.value)


def test_chunk_record_vectors_must_be_numeric() -> None:
    with pytest.raises(ValueError) as exc_info:
        ChunkRecord(
            id="record-1",
            text="x",
            metadata=_metadata(),
            dense_vector=[1.0, "bad"],
        )
    assert "dense_vector must contain numeric values" in str(exc_info.value)
