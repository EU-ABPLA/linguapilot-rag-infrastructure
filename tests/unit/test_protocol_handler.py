from __future__ import annotations

from typing import Any, Dict, Mapping

from mcp_server.protocol_handler import ProtocolHandler


def test_initialize_returns_server_info_and_capabilities() -> None:
    handler = ProtocolHandler()
    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        }
    )
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    result = response["result"]
    assert result["serverInfo"]["name"] == "linguapilot-mcp-server"
    assert result["protocolVersion"] == "2024-11-05"
    assert "tools" in result["capabilities"]


def test_tools_list_returns_registered_schemas() -> None:
    handler = ProtocolHandler(
        tool_schemas=[
            {
                "name": "query_knowledge_hub",
                "description": "query",
                "inputSchema": {"type": "object"},
            }
        ]
    )
    response = handler.handle_request(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    )
    tools = response["result"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "query_knowledge_hub"


def test_tools_call_routes_and_returns_result() -> None:
    called: Dict[str, Any] = {}

    def tool(arguments: Mapping[str, Any]) -> Dict[str, Any]:
        called["arguments"] = dict(arguments)
        return {"ok": True, "echo": arguments.get("q")}

    handler = ProtocolHandler(tool_handlers={"query_knowledge_hub": tool})
    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "query_knowledge_hub", "arguments": {"q": "hello"}},
        }
    )
    assert response["result"]["ok"] is True
    assert response["result"]["echo"] == "hello"
    assert called["arguments"]["q"] == "hello"


def test_invalid_method_returns_method_not_found() -> None:
    handler = ProtocolHandler()
    response = handler.handle_request(
        {"jsonrpc": "2.0", "id": 4, "method": "unknown/method", "params": {}}
    )
    assert response["error"]["code"] == -32601


def test_invalid_params_return_invalid_params_error() -> None:
    handler = ProtocolHandler(tool_handlers={"x": lambda args: args})
    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "x", "arguments": "bad"},
        }
    )
    assert response["error"]["code"] == -32602


def test_internal_exception_returns_internal_error() -> None:
    def crash(arguments: Mapping[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("boom")

    handler = ProtocolHandler(tool_handlers={"x": crash})
    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "x", "arguments": {}},
        }
    )
    assert response["error"]["code"] == -32603


def test_parse_error_maps_to_jsonrpc_parse_error() -> None:
    handler = ProtocolHandler()
    response = handler.handle_json_message("{bad json")
    assert response["error"]["code"] == -32700
