from dataclasses import dataclass
from typing import Any, List, Optional

from core import Chunk, Document
from ingestion.chunking import DocumentChunker
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


@dataclass
class SplitterConfig:
    provider: str
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass
class Settings:
    splitter: SplitterConfig


class FakeSplitter(BaseSplitter):
    def __init__(self, outputs: List[str]):
        self._outputs = list(outputs)

    def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
        return list(self._outputs)


def test_document_chunker_converts_text_to_chunk_contract() -> None:
    provider = "doc_chunker_fake_basic"
    SplitterFactory.register(provider, lambda settings: FakeSplitter(["Alpha", "Beta"]))
    try:
        chunker = DocumentChunker(Settings(splitter=SplitterConfig(provider=provider)))
        document = Document(
            id="doc-1",
            text="Alpha\n\nBeta",
            metadata={"source_path": "data/docs/a.pdf", "doc_type": "pdf"},
        )
        chunks = chunker.split_document(document)
    finally:
        SplitterFactory.unregister(provider)

    assert len(chunks) == 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert chunks[0].id.startswith("doc-1_0000_")
    assert chunks[1].id.startswith("doc-1_0001_")
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1
    assert chunks[0].source_ref == "doc-1"
    assert chunks[1].source_ref == "doc-1"
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == 5
    assert chunks[1].start_offset > chunks[0].start_offset
    assert chunks[1].end_offset > chunks[1].start_offset


def test_document_chunker_ids_are_unique_and_deterministic() -> None:
    provider = "doc_chunker_fake_deterministic"
    SplitterFactory.register(provider, lambda settings: FakeSplitter(["One", "Two", "Two"]))
    try:
        chunker = DocumentChunker(Settings(splitter=SplitterConfig(provider=provider)))
        document = Document(
            id="doc-2",
            text="One\nTwo\nTwo",
            metadata={"source_path": "data/docs/b.pdf"},
        )
        first = chunker.split_document(document)
        second = chunker.split_document(document)
    finally:
        SplitterFactory.unregister(provider)

    first_ids = [chunk.id for chunk in first]
    second_ids = [chunk.id for chunk in second]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))


def test_document_chunker_distributes_images_per_chunk() -> None:
    provider = "doc_chunker_fake_images"
    SplitterFactory.register(
        provider,
        lambda settings: FakeSplitter(
            [
                "Intro [IMAGE: img_1]",
                "No image here",
                "Tail [IMAGE: img_2] [IMAGE: img_1] [IMAGE: missing]",
            ]
        ),
    )
    try:
        chunker = DocumentChunker(Settings(splitter=SplitterConfig(provider=provider)))
        document = Document(
            id="doc-3",
            text="Intro [IMAGE: img_1]\nNo image here\nTail [IMAGE: img_2] [IMAGE: img_1] [IMAGE: missing]",
            metadata={
                "source_path": "data/docs/c.pdf",
                "images": [
                    {
                        "id": "img_1",
                        "path": "data/images/doc3/img_1.png",
                        "page": 1,
                        "text_offset": 0,
                        "text_length": 16,
                        "position": {"x": 0, "y": 0},
                    },
                    {
                        "id": "img_2",
                        "path": "data/images/doc3/img_2.png",
                        "page": 1,
                        "text_offset": 33,
                        "text_length": 16,
                        "position": {"x": 1, "y": 1},
                    },
                ],
            },
        )
        chunks = chunker.split_document(document)
    finally:
        SplitterFactory.unregister(provider)

    assert chunks[0].metadata["image_refs"] == ["img_1"]
    assert [item["id"] for item in chunks[0].metadata["images"]] == ["img_1"]
    assert chunks[1].metadata["image_refs"] == []
    assert "images" not in chunks[1].metadata
    assert chunks[2].metadata["image_refs"] == ["img_2", "img_1", "missing"]
    assert [item["id"] for item in chunks[2].metadata["images"]] == ["img_2", "img_1"]
    assert [item["id"] for item in document.metadata["images"]] == ["img_1", "img_2"]


def test_document_chunker_is_config_driven_by_splitter_settings() -> None:
    document = Document(
        id="doc-4",
        text="# T\n\n" + ("A" * 180) + "\n\n## B\n" + ("C" * 180),
        metadata={"source_path": "data/docs/d.pdf"},
    )
    small_settings = {"splitter": {"provider": "recursive", "chunk_size": 50, "chunk_overlap": 0}}
    large_settings = {"splitter": {"provider": "recursive", "chunk_size": 400, "chunk_overlap": 0}}
    small_chunks = DocumentChunker(small_settings).split_document(document)
    large_chunks = DocumentChunker(large_settings).split_document(document)

    assert len(small_chunks) > len(large_chunks)
