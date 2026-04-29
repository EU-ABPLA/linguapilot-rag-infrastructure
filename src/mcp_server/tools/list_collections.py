from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping


def get_tool_schema() -> Dict[str, Any]:
    return {
        "name": "list_collections",
        "description": "List available document collections.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    }


def call_list_collections(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    return list_collections()


def list_collections(root_dir: str = "data/documents") -> Dict[str, Any]:
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        return {"collections": []}
    collections: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        document_count = 0
        for item in child.rglob("*"):
            if item.is_file():
                document_count += 1
        collections.append(
            {
                "name": child.name,
                "document_count": document_count,
            }
        )
    return {"collections": collections}
