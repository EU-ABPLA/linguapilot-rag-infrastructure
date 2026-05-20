from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple

import pytest


def _page_targets() -> List[Tuple[str, str]]:
    return [
        ("overview", "observability.dashboard.pages.overview"),
        ("data_browser", "observability.dashboard.pages.data_browser"),
        ("ingestion_manager", "observability.dashboard.pages.ingestion_manager"),
        ("ingestion_traces", "observability.dashboard.pages.ingestion_traces"),
        ("query_traces", "observability.dashboard.pages.query_traces"),
        ("evaluation_panel", "observability.dashboard.pages.evaluation_panel"),
    ]


@pytest.mark.e2e
@pytest.mark.parametrize(("page_name", "module_name"), _page_targets())
def test_dashboard_pages_render_without_exception(
    page_name: str,
    module_name: str,
) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    script_path = _write_page_script(module_name)
    try:
        app_test = streamlit_testing.AppTest.from_file(str(script_path))
        app_test.run(timeout=20)
        assert len(app_test.exception) == 0, page_name
    finally:
        script_path.unlink(missing_ok=True)


def _write_page_script(module_name: str) -> Path:
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
        handle.write("from importlib import import_module\n")
        handle.write("module = import_module(" + repr(module_name) + ")\n")
        handle.write("module.render_page()\n")
        return Path(handle.name)
