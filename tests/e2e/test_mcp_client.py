from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def test_mcp_client_roundtrip_list_and_call() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        cwd=str(repo_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        init = _send_jsonrpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest-mcp-client", "version": "1.0"},
                },
            },
        )
        assert init["jsonrpc"] == "2.0"
        assert init["id"] == 1
        assert init["result"]["serverInfo"]["name"] == "linguapilot-mcp-server"
        listing = _send_jsonrpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            },
        )
        tools = listing["result"]["tools"]
        names = [item["name"] for item in tools]
        assert "query_knowledge_hub" in names
        called = _send_jsonrpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {
                        "query": "How to setup Azure OpenAI endpoint?",
                        "top_k": 3,
                    },
                },
            },
        )
        assert called["jsonrpc"] == "2.0"
        assert called["id"] == 3
        assert "result" in called
        payload = called["result"]
        assert isinstance(payload.get("content"), list)
        assert "structuredContent" in payload
        structured = payload["structuredContent"]
        assert isinstance(structured.get("citations"), list)
        assert structured.get("query") == "How to setup Azure OpenAI endpoint?"
        if proc.stdin is not None:
            proc.stdin.write(
                json.dumps(
                    {"jsonrpc": "2.0", "method": "exit"},
                    ensure_ascii=True,
                )
                + "\n"
            )
            proc.stdin.flush()
    finally:
        _close_stdin(proc)
        proc.wait(timeout=10)
    assert proc.returncode == 0


def _send_jsonrpc(proc: subprocess.Popen[str], payload: Dict[str, Any]) -> Dict[str, Any]:
    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("stdio pipe is not available")
    proc.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        stderr_text = ""
        if proc.stderr is not None:
            stderr_text = proc.stderr.read()
        raise RuntimeError("empty response from server: " + stderr_text)
    parsed = json.loads(line)
    if not isinstance(parsed, dict):
        raise RuntimeError("invalid jsonrpc response type")
    return parsed


def _close_stdin(proc: subprocess.Popen[str]) -> None:
    if proc.stdin is not None:
        proc.stdin.close()
