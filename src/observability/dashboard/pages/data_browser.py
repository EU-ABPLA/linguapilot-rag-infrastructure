from __future__ import annotations

from observability.dashboard.services.data_service import DataService


def render_page() -> None:
    import streamlit as st

    st.header("Data Browser")
    service = DataService()
    collections = service.list_collections()
    collection_options = ["all"] + collections
    selected_collection = st.selectbox("Collection", collection_options, index=0)
    active_collection = None if selected_collection == "all" else selected_collection
    documents = service.list_documents(collection=active_collection)
    if not documents:
        st.info("No ingested documents found.")
        return
    summary_rows = []
    for item in documents:
        summary_rows.append(
            {
                "source_path": item["source_path"],
                "collection": item["collection"],
                "chunk_count": item["chunk_count"],
                "image_count": item["image_count"],
                "status": item["status"],
                "processed_at": item["processed_at"],
            }
        )
    st.subheader("Documents")
    st.dataframe(summary_rows, use_container_width=True)
    st.subheader("Document Details")
    for item in documents:
        title = item["source_path"] + " (" + str(item["chunk_count"]) + " chunks)"
        with st.expander(title):
            detail = service.get_document_detail(
                source_path=item["source_path"],
                collection=item["collection"],
            )
            st.write("Collection: " + str(detail["collection"]))
            st.write("Status: " + str(detail["status"]))
            st.write("Processed At: " + str(detail["processed_at"]))
            st.write("Image Count: " + str(len(detail["images"])))
            st.markdown("### Chunks")
            chunk_rows = []
            for chunk in detail["chunks"]:
                chunk_rows.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    }
                )
            if chunk_rows:
                st.dataframe(chunk_rows, use_container_width=True)
            if detail["images"]:
                st.markdown("### Images")
                for image in detail["images"]:
                    image_path = str(image.get("file_path", ""))
                    st.write("Image ID: " + str(image.get("image_id", "")))
                    st.write("Page: " + str(image.get("page_num", "")))
                    if image_path:
                        st.image(image_path, caption=str(image.get("image_id", "")))
