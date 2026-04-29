from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

from core.response.response_builder import ResponseBuilder
from core.types import RetrievalResult


def test_mcp_server_initialize_and_stdio_constraints() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "1.0"},
        },
    }
    proc = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request, ensure_ascii=True) + "\n",
        text=True,
        capture_output=True,
        cwd=str(repo_root),
        check=False,
    )
    assert proc.returncode == 0
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) >= 1
    parsed = [json.loads(line) for line in stdout_lines]
    response = parsed[0]
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    result = response["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["serverInfo"]["name"] == "linguapilot-mcp-server"
    assert "tools" in result["capabilities"]
    stderr_text = proc.stderr
    assert "mcp server started" in stderr_text
    assert "initialize request received" in stderr_text


def test_mcp_server_tools_list_contains_query_tool() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    proc = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request, ensure_ascii=True) + "\n",
        text=True,
        capture_output=True,
        cwd=str(repo_root),
        check=False,
    )
    assert proc.returncode == 0
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    response = json.loads(stdout_lines[0])
    tools = response["result"]["tools"]
    names = [item["name"] for item in tools]
    assert "query_knowledge_hub" in names


def test_mcp_server_query_tool_returns_structured_payload() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "query_knowledge_hub",
            "arguments": {"query": "如何配置 Azure？", "top_k": 3},
        },
    }
    proc = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request, ensure_ascii=True) + "\n",
        text=True,
        capture_output=True,
        cwd=str(repo_root),
        check=False,
    )
    assert proc.returncode == 0
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    response = json.loads(stdout_lines[0])
    assert "result" in response
    result = response["result"]
    assert "content" in result
    assert "structuredContent" in result
    assert isinstance(result["content"], list)
    assert isinstance(result["structuredContent"]["citations"], list)


def test_mcp_response_contains_image_content_when_chunk_has_images(tmp_path: Path) -> None:
    image_path = tmp_path / "hit.png"
    image_bytes = b"\x89PNG\r\n\x1a\nmultimodal-test"
    image_path.write_bytes(image_bytes)
    result = RetrievalResult(
        chunk_id="chunk-image-1",
        score=0.95,
        text="This chunk references an image.",
        metadata={
            "source_path": "docs/image.md",
            "image_refs": ["img-1"],
            "images": [{"id": "img-1", "path": str(image_path)}],
        },
    )
    payload = ResponseBuilder().build([result], "show me image")
    content = payload["content"]
    assert content[0]["type"] == "text"
    image_items = [item for item in content if item.get("type") == "image"]
    assert len(image_items) == 1
    image_item = image_items[0]
    assert image_item["mimeType"] == "image/png"
    decoded = base64.b64decode(image_item["data"])
    assert decoded == image_bytes
