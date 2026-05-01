from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

PageEntry = Tuple[str, Callable[[], None]]


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="LinguaPilot Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("LinguaPilot Dashboard")
    entries = _page_entries()
    if hasattr(st, "navigation"):
        pages = [st.Page(fn, title=title) for title, fn in entries]
        navigation = st.navigation(pages)
        navigation.run()
        return
    labels = [item[0] for item in entries]
    selected = st.sidebar.selectbox("Pages", labels, index=0)
    index = labels.index(selected)
    entries[index][1]()


def _page_entries() -> Sequence[PageEntry]:
    from observability.dashboard.pages import (
        data_browser,
        evaluation_panel,
        ingestion_manager,
        ingestion_traces,
        overview,
        query_traces,
    )

    return [
        ("Overview", overview.render_page),
        ("Data Browser", data_browser.render_page),
        ("Ingestion Manager", ingestion_manager.render_page),
        ("Ingestion Traces", ingestion_traces.render_page),
        ("Query Traces", query_traces.render_page),
        ("Evaluation Panel", evaluation_panel.render_page),
    ]


if __name__ == "__main__":
    main()
