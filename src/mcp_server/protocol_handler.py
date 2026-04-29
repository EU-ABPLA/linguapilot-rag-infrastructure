from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

ToolHandler = Callable[[Mapping[str, Any]], Any]


class ProtocolHandler:
    def __init__(
        self,
        tool_handlers: Optional[Mapping[str, ToolHandler]] = None,
        tool_schemas: Optional[Sequence[Mapping[str, Any]]] = None,
        server_name: str = "linguapilot-mcp-server",
        server_version: str = "0.1.0",
        protocol_version: str = "2024-11-05",
    ):
        self._tool_handlers = dict(tool_handlers or {})
        self._tool_schemas = [dict(item) for item in (tool_schemas or [])]
        self._server_name = server_name
        self._server_version = server_version
        self._protocol_version = protocol_version

    def handle_json_message(self, raw: str) -> Dict[str, Any]:
        try:
            request = json.loads(raw)
        except Exception:
            return _error_response(None, -32700, "Parse error")
        return self.handle_request(request)

    def handle_request(self, request: Any) -> Dict[str, Any]:
        if not isinstance(request, Mapping):
            return _error_response(None, -32600, "Invalid Request")
        request_id = request.get("id")
        jsonrpc = request.get("jsonrpc")
        method = request.get("method")
        if jsonrpc != "2.0" or not isinstance(method, str) or not method.strip():
            return _error_response(request_id, -32600, "Invalid Request")
        params = request.get("params", {})
        try:
            result = self._dispatch(method.strip(), params)
        except _JsonRpcError as exc:
            return _error_response(request_id, exc.code, exc.message)
        except Exception:
            return _error_response(request_id, -32603, "Internal error")
        return _ok_response(request_id, result)

    def _dispatch(self, method: str, params: Any) -> Any:
        if method == "initialize":
            return self.handle_initialize(params)
        if method == "tools/list":
            return self.handle_tools_list()
        if method == "tools/call":
            if not isinstance(params, Mapping):
                raise _JsonRpcError(-32602, "Invalid params")
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str) or not name.strip():
                raise _JsonRpcError(-32602, "Invalid params")
            return self.handle_tools_call(name.strip(), arguments)
        raise _JsonRpcError(-32601, "Method not found")

    def handle_initialize(self, params: Any) -> Dict[str, Any]:
        protocol_version = self._protocol_version
        if isinstance(params, Mapping):
            raw = params.get("protocolVersion")
            if isinstance(raw, str) and raw.strip():
                protocol_version = raw.strip()
        return {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": self._server_name,
                "version": self._server_version,
            },
        }

    def handle_tools_list(self) -> Dict[str, Any]:
        if self._tool_schemas:
            return {"tools": [dict(item) for item in self._tool_schemas]}
        tools: List[Dict[str, Any]] = []
        for name in sorted(self._tool_handlers.keys()):
            tools.append(
                {
                    "name": name,
                    "description": "",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            )
        return {"tools": tools}

    def handle_tools_call(self, name: str, arguments: Any) -> Any:
        handler = self._tool_handlers.get(name)
        if handler is None:
            raise _JsonRpcError(-32601, "Method not found")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise _JsonRpcError(-32602, "Invalid params")
        try:
            return handler(dict(arguments))
        except _JsonRpcError:
            raise
        except (ValueError, TypeError):
            raise _JsonRpcError(-32602, "Invalid params")
        except Exception:
            raise _JsonRpcError(-32603, "Internal error")


class _JsonRpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _ok_response(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": int(code), "message": message},
    }
