from __future__ import annotations

from observability.dashboard.services.config_service import ConfigService


def render_page() -> None:
    import streamlit as st

    st.header("System Overview")
    service = ConfigService()
    components = service.get_component_cards()
    stats = service.get_collection_stats()
    st.subheader("Component Configuration")
    if not components:
        st.info("No component configuration available.")
    else:
        columns = st.columns(2)
        for index, item in enumerate(components):
            column = columns[index % 2]
            with column:
                st.markdown("### " + item["name"])
                st.write("Provider: " + item["provider"])
                if item.get("model"):
                    st.write("Model: " + str(item.get("model")))
                st.write("Enabled: " + str(item.get("enabled")))
                details = item.get("details", {})
                if details:
                    st.json(details)
    st.subheader("Collection Statistics")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Collection", str(stats.get("collection", "default")))
    metric_columns[1].metric("Vector Records", str(stats.get("vector_records", 0)))
    metric_columns[2].metric("Unique Sources", str(stats.get("unique_sources", 0)))
