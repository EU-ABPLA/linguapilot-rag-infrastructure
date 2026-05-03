from __future__ import annotations

from typing import Callable, List, Tuple

import pytest


def _page_targets() -> List[Tuple[str, Callable[[], None]]]:
    from observability.dashboard.pages import (
        data_browser,
        evaluation_panel,
        ingestion_manager,
        ingestion_traces,
        overview,
        query_traces,
    )

    return [
        ("overview", overview.render_page),
        ("data_browser", data_browser.render_page),
        ("ingestion_manager", ingestion_manager.render_page),
        ("ingestion_traces", ingestion_traces.render_page),
        ("query_traces", query_traces.render_page),
        ("evaluation_panel", evaluation_panel.render_page),
    ]


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("page_name", "render_fn"),
    _page_targets(),
)
def test_dashboard_pages_render_without_exception(
    page_name: str,
    render_fn: Callable[[], None],
) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app_test = streamlit_testing.AppTest.from_function(render_fn)
    app_test.run(timeout=20)
    assert len(app_test.exception) == 0, page_name
