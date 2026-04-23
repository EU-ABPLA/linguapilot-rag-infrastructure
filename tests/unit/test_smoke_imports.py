import importlib


def test_top_level_packages_importable() -> None:
    modules = ["mcp_server", "core", "ingestion", "libs", "observability"]
    for name in modules:
        imported = importlib.import_module(name)
        assert imported is not None
