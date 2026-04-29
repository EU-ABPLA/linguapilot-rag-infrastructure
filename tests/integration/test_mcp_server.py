from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
