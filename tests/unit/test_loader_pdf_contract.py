from pathlib import Path

from core.types import Document
from libs.loader.pdf_loader import PdfLoader


def test_pdf_loader_returns_document_with_source_path(tmp_path: Path) -> None:
    pdf_path = tmp_path / "simple.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n(Hello PDF)\n%%EOF")
    loader = PdfLoader(image_output_root=str(tmp_path / "images"))
    document = loader.load(str(pdf_path))
    assert isinstance(document, Document)
    assert document.metadata["source_path"] == str(pdf_path)
    assert document.metadata["doc_type"] == "pdf"
    assert isinstance(document.text, str)
    assert len(document.text) > 0


def test_pdf_loader_extracts_image_placeholders_and_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "with_images.pdf"
    pdf_path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj\n/Subtype /Image\nendobj\n"
        b"2 0 obj\n/Subtype /Image\nendobj\n"
        b"%%EOF"
    )
    loader = PdfLoader(image_output_root=str(tmp_path / "images"))
    document = loader.load(str(pdf_path))
    images = document.metadata.get("images")
    assert isinstance(images, list)
    assert len(images) == 2
    for image in images:
        assert Path(image["path"]).exists()
        assert image["text_length"] > 0
        assert image["page"] == 1
        placeholder = "[IMAGE: " + image["id"] + "]"
        assert placeholder in document.text
