from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from core.settings import SettingsError, load_settings
from ingestion.document_manager import DocumentManager
from ingestion.pipeline import IngestionPipeline
from libs.vector_store.chroma_store import ChromaStore


def render_page() -> None:
    import streamlit as st

    st.header("Ingestion Manager")
    collection = st.text_input("Collection", value="default")
    force = st.checkbox("Force Reingest", value=False)
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    progress = st.progress(0.0)
    status_area = st.empty()
    result_area = st.container()
    if st.button("Start Ingestion", disabled=not uploaded_files):
        normalized_collection = _normalize_collection(collection)
        upload_root = Path("data/documents") / normalized_collection
        upload_root.mkdir(parents=True, exist_ok=True)
        try:
            settings = load_settings("config/settings.yaml")
        except SettingsError as exc:
            st.error(str(exc))
            return
        pipeline = IngestionPipeline(settings)
        total_files = len(uploaded_files or [])
        completed = 0
        result_rows: List[Dict[str, Any]] = []
        for uploaded in uploaded_files or []:
            file_name = _safe_filename(uploaded.name)
            target_path = _next_available_file(upload_root / file_name)
            target_path.write_bytes(bytes(uploaded.getbuffer()))
            stage_state = {"stage": "pending", "current": 0, "total": 1}

            def _on_progress(stage: str, current: int, total: int) -> None:
                stage_state["stage"] = stage
                stage_state["current"] = current
                stage_state["total"] = total
                ratio = float(current) / float(total) if total > 0 else 0.0
                overall = (float(completed) + max(0.0, min(1.0, ratio))) / float(
                    max(total_files, 1)
                )
                progress.progress(max(0.0, min(1.0, overall)))
                status_area.write(
                    "Processing "
                    + file_name
                    + " | stage="
                    + str(stage)
                    + " "
                    + str(current)
                    + "/"
                    + str(total)
                )

            try:
                result = pipeline.run(
                    str(target_path),
                    collection=normalized_collection,
                    force=force,
                    on_progress=_on_progress,
                )
                result_rows.append(
                    {
                        "file": file_name,
                        "status": result.status,
                        "chunks": result.chunk_count,
                        "vectors": result.vector_count,
                        "images": result.image_count,
                    }
                )
            except Exception as exc:
                result_rows.append(
                    {
                        "file": file_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
            completed += 1
            progress.progress(float(completed) / float(max(total_files, 1)))
        status_area.write("Ingestion completed.")
        with result_area:
            st.subheader("Run Results")
            st.dataframe(result_rows, use_container_width=True)
    st.subheader("Existing Documents")
    normalized_collection = _normalize_collection(collection)
    manager = DocumentManager(chroma_store=ChromaStore(collection=normalized_collection))
    docs = manager.list_documents(collection=normalized_collection)
    if not docs:
        st.info("No documents found in collection: " + normalized_collection)
        return
    for item in docs:
        title = item.source_path + " (" + str(item.chunk_count) + " chunks)"
        with st.expander(title):
            st.write("Collection: " + item.collection)
            st.write("Images: " + str(item.image_count))
            if st.button(
                "Delete Document",
                key="delete_" + _safe_key(item.source_path) + "_" + item.collection,
            ):
                try:
                    deleted = manager.delete_document(
                        source_path=item.source_path,
                        collection=item.collection,
                    )
                    if deleted.success:
                        st.success("Deleted: " + item.source_path)
                    else:
                        st.warning("No records deleted for: " + item.source_path)
                except Exception as exc:
                    st.error(str(exc))


def _normalize_collection(value: str) -> str:
    stripped = value.strip()
    if stripped:
        return stripped
    return "default"


def _safe_filename(value: str) -> str:
    name = Path(str(value)).name.strip()
    if name:
        return name
    return "uploaded.pdf"


def _safe_key(value: str) -> str:
    output = []
    for ch in value:
        if ch.isalnum():
            output.append(ch)
        else:
            output.append("_")
    return "".join(output)


def _next_available_file(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / (stem + "_" + str(index) + suffix)
        if not candidate.exists():
            return candidate
        index += 1
