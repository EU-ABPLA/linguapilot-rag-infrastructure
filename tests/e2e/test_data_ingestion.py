from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

from ingestion.pipeline import IngestionResult


def _load_ingest_module() -> object:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "ingest.py"
    spec = importlib.util.spec_from_file_location("ingest_script_for_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ingest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.4\n1 0 obj\n(hello)\n%%EOF")


class _FakePipeline:
    _processed = set()

    def __init__(self, settings: object):
        self._settings = settings

    def run(
        self,
        source_path: str,
        collection: str = "default",
        force: bool = False,
        on_progress: Optional[object] = None,
        trace: Optional[object] = None,
    ) -> IngestionResult:
        key = collection + ":" + source_path
        if not force and key in self.__class__._processed:
            return IngestionResult(
                status="skipped",
                source_path=source_path,
                collection=collection,
                file_hash="fake_hash",
                skipped=True,
                document_id=None,
                chunk_count=0,
                vector_count=0,
                image_count=0,
                bm25_doc_count=0,
            )
        self.__class__._processed.add(key)
        db_root = Path("data/db")
        (db_root / "chroma").mkdir(parents=True, exist_ok=True)
        (db_root / "bm25").mkdir(parents=True, exist_ok=True)
        (Path("data/images") / collection).mkdir(parents=True, exist_ok=True)
        (db_root / "chroma" / (collection + ".json")).write_text("{}", encoding="utf-8")
        (db_root / "bm25" / "bm25_index.pkl").write_bytes(b"bm25")
        return IngestionResult(
            status="success",
            source_path=source_path,
            collection=collection,
            file_hash="fake_hash",
            skipped=False,
            document_id="doc-1",
            chunk_count=2,
            vector_count=2,
            image_count=1,
            bm25_doc_count=2,
        )


def test_ingest_script_generates_artifacts_and_skips_second_run(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _FakePipeline._processed = set()
    module = _load_ingest_module()
    monkeypatch.setattr(module, "IngestionPipeline", _FakePipeline)
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "simple.pdf"
    _write_pdf(source)
    config_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    first_exit = module.main(
        [
            "--path",
            str(source),
            "--collection",
            "default",
            "--config",
            str(config_path),
        ]
    )
    first_output = capsys.readouterr().out
    assert first_exit == 0
    assert "status=success" in first_output
    assert (tmp_path / "data/db/chroma/default.json").exists()
    assert (tmp_path / "data/db/bm25/bm25_index.pkl").exists()

    second_exit = module.main(
        [
            "--path",
            str(source),
            "--collection",
            "default",
            "--config",
            str(config_path),
        ]
    )
    second_output = capsys.readouterr().out
    assert second_exit == 0
    assert "status=skipped" in second_output


def test_ingest_script_force_reingests_even_if_processed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _FakePipeline._processed = set()
    module = _load_ingest_module()
    monkeypatch.setattr(module, "IngestionPipeline", _FakePipeline)
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "simple.pdf"
    _write_pdf(source)
    config_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    module.main(
        [
            "--path",
            str(source),
            "--collection",
            "default",
            "--config",
            str(config_path),
        ]
    )
    capsys.readouterr()
    forced_exit = module.main(
        [
            "--path",
            str(source),
            "--collection",
            "default",
            "--force",
            "--config",
            str(config_path),
        ]
    )
    forced_output = capsys.readouterr().out
    assert forced_exit == 0
    assert "status=success" in forced_output
