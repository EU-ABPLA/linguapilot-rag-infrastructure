from __future__ import annotations

import copy
import hashlib
import re
from typing import Dict, List, Mapping, Tuple

from core.settings import Settings
from core.types import Chunk, Document
from libs.splitter.splitter_factory import SplitterFactory

_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]\s]+)\s*\]")


class DocumentChunker:
    def __init__(self, settings: Settings):
        self._splitter = SplitterFactory.create(settings)

    def split_document(self, document: Document) -> List[Chunk]:
        if not isinstance(document, Document):
            raise ValueError("document must be a Document")
        raw_chunks = self._splitter.split_text(document.text)
        chunks: List[Chunk] = []
        search_start = 0
        for chunk_text in raw_chunks:
            if not isinstance(chunk_text, str):
                raise ValueError("splitter output must be List[str]")
            normalized_text = chunk_text.strip()
            if not normalized_text:
                continue
            chunk_index = len(chunks)
            start_offset, end_offset, search_start = _locate_offsets(
                document.text, normalized_text, search_start
            )
            chunks.append(
                Chunk(
                    id=self._generate_chunk_id(document.id, chunk_index, normalized_text),
                    text=normalized_text,
                    metadata=self._inherit_metadata(
                        document=document,
                        chunk_index=chunk_index,
                        chunk_text=normalized_text,
                    ),
                    start_offset=start_offset,
                    end_offset=end_offset,
                    source_ref=document.id,
                )
            )
        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        digest = hashlib.sha256(
            (doc_id + "\n" + str(index) + "\n" + text).encode("utf-8")
        ).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{digest}"

    def _inherit_metadata(
        self, document: Document, chunk_index: int, chunk_text: str
    ) -> Dict[str, object]:
        metadata = copy.deepcopy(document.metadata)
        metadata["chunk_index"] = chunk_index
        refs = _extract_image_refs(chunk_text)
        all_images = document.metadata.get("images")
        if refs:
            metadata["image_refs"] = refs
            selected_images = _select_image_subset(all_images, refs)
            if selected_images:
                metadata["images"] = selected_images
            else:
                metadata.pop("images", None)
        else:
            metadata.pop("images", None)
            metadata["image_refs"] = []
        return metadata


def _locate_offsets(
    document_text: str, chunk_text: str, search_start: int
) -> Tuple[int, int, int]:
    start_offset = document_text.find(chunk_text, search_start)
    if start_offset < 0:
        start_offset = document_text.find(chunk_text)
    if start_offset < 0:
        start_offset = min(max(search_start, 0), len(document_text))
    end_offset = min(start_offset + len(chunk_text), len(document_text))
    next_search = max(search_start + 1, end_offset)
    return start_offset, end_offset, next_search


def _extract_image_refs(chunk_text: str) -> List[str]:
    refs: List[str] = []
    for match in _IMAGE_PLACEHOLDER_PATTERN.finditer(chunk_text):
        image_id = match.group(1)
        if image_id not in refs:
            refs.append(image_id)
    return refs


def _select_image_subset(all_images: object, refs: List[str]) -> List[Dict[str, object]]:
    if not isinstance(all_images, list):
        return []
    image_map: Dict[str, Mapping[str, object]] = {}
    for image in all_images:
        if not isinstance(image, Mapping):
            continue
        image_id = image.get("id")
        if isinstance(image_id, str) and image_id:
            image_map[image_id] = image
    selected: List[Dict[str, object]] = []
    for ref in refs:
        value = image_map.get(ref)
        if value is not None:
            selected.append(copy.deepcopy(dict(value)))
    return selected
