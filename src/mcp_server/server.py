from __future__ import annotations

import json
import sys
from typing import Any, Dict, Mapping, Optional

from mcp_server.protocol_handler import ProtocolHandler
from mcp_server.tools.query_knowledge_hub import (
    call_query_knowledge_hub,
    get_tool_schema,
)
from observability.logger import get_logger

_JSONRPC_VERSION = "2.0"
_SERVER_NAME = "linguapilot-mcp-server"
_SERVER_VERSION = "0.1.0"


def main() -> int:
    logger = get_logger("mcp_server.server")
    protocol_handler = ProtocolHandler(
        tool_handlers={"query_knowledge_hub": call_query_knowledge_hub},
        tool_schemas=[get_tool_schema()],
        server_name=_SERVER_NAME,
        server_version=_SERVER_VERSION,
    )
    logger.info("mcp server started")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        response = _handle_line(line, logger=logger, protocol_handler=protocol_handler)
        if response is None:
            continue
        _write_response(response)
    logger.info("mcp server stopped")
    return 0


def _handle_line(
    line: str,
    logger: Any,
    protocol_handler: ProtocolHandler,
) -> Optional[Dict[str, Any]]:
    try:
        request = json.loads(line)
    except Exception:
        return _error_response(None, code=-32700, message="Parse error")
    if not isinstance(request, Mapping):
        return _error_response(None, code=-32600, message="Invalid Request")
    method = request.get("method")
    if method == "initialize":
        logger.info("initialize request received")
    if method == "shutdown":
        return protocol_handler.handle_request(request)
    if method == "exit":
        return None
    return protocol_handler.handle_request(request)


def _initialize_result(params: Any) -> Dict[str, Any]:
    protocol_version = "2024-11-05"
    if isinstance(params, Mapping):
        raw = params.get("protocolVersion")
        if isinstance(raw, str) and raw.strip():
            protocol_version = raw.strip()
    return {
        "protocolVersion": protocol_version,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": _SERVER_NAME, "version": _SERVER_VERSION},
    }


def _ok_response(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": _JSONRPC_VERSION, "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": _JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": int(code), "message": message},
    }


def _write_response(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(dict(payload), ensure_ascii=True) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
