from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

PageEntry = Tuple[str, str, Callable[[], None]]

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


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
        pages = [st.Page(fn, title=title, url_path=url_path) for title, url_path, fn in entries]
        navigation = st.navigation(pages)
        navigation.run()
        return
    labels = [item[0] for item in entries]
    selected = st.sidebar.selectbox("Pages", labels, index=0)
    index = labels.index(selected)
    entries[index][2]()


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
        ("Overview", "overview", overview.render_page),
        ("Data Browser", "data-browser", data_browser.render_page),
        ("Ingestion Manager", "ingestion-manager", ingestion_manager.render_page),
        ("Ingestion Traces", "ingestion-traces", ingestion_traces.render_page),
        ("Query Traces", "query-traces", query_traces.render_page),
        ("Evaluation Panel", "evaluation-panel", evaluation_panel.render_page),
    ]


if __name__ == "__main__":
    main()
