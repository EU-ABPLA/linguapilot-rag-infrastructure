from pathlib import Path

from mcp_server.tools.list_collections import list_collections


def test_list_collections_returns_sorted_names_with_counts(tmp_path: Path) -> None:
    root = tmp_path / "data" / "documents"
    alpha = root / "alpha"
    beta = root / "beta"
    alpha.mkdir(parents=True, exist_ok=True)
    beta.mkdir(parents=True, exist_ok=True)
    (alpha / "a.md").write_text("a", encoding="utf-8")
    (alpha / "b.md").write_text("b", encoding="utf-8")
    (beta / "x.md").write_text("x", encoding="utf-8")
    payload = list_collections(str(root))
    assert "collections" in payload
    items = payload["collections"]
    assert [item["name"] for item in items] == ["alpha", "beta"]
    assert items[0]["document_count"] == 2
    assert items[1]["document_count"] == 1


def test_list_collections_returns_empty_when_root_missing(tmp_path: Path) -> None:
    payload = list_collections(str(tmp_path / "missing" / "documents"))
    assert payload == {"collections": []}
