from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

"""
LLM client will call the server with a base verb, and the server will return the past, present, and future tense forms of that verb.

ensure server :
python week3.py --serve

run demo : 
python week3.py --demo --base-verb run

"""

JSONRPC_VERSION = "2.0"
SERVER_NAME = "week3-verb-mcp-server"
SERVER_VERSION = "0.1.0"

_IRREGULAR_VERBS: Dict[str, Dict[str, str]] = {
    "be": {"past": "was/were", "present": "be", "future": "will be"},
    "have": {"past": "had", "present": "have", "future": "will have"},
    "do": {"past": "did", "present": "do", "future": "will do"},
    "go": {"past": "went", "present": "go", "future": "will go"},
    "get": {"past": "got", "present": "get", "future": "will get"},
    "make": {"past": "made", "present": "make", "future": "will make"},
    "know": {"past": "knew", "present": "know", "future": "will know"},
    "think": {"past": "thought", "present": "think", "future": "will think"},
    "take": {"past": "took", "present": "take", "future": "will take"},
    "see": {"past": "saw", "present": "see", "future": "will see"},
    "come": {"past": "came", "present": "come", "future": "will come"},
    "say": {"past": "said", "present": "say", "future": "will say"},
    "give": {"past": "gave", "present": "give", "future": "will give"},
    "run": {"past": "ran", "present": "run", "future": "will run"},
    "tell": {"past": "told", "present": "tell", "future": "will tell"},
    "find": {"past": "found", "present": "find", "future": "will find"},
    "work": {"past": "worked", "present": "work", "future": "will work"},
}
_VOWELS = set("aeiou")


def get_tool_schema() -> Dict[str, Any]:
    return {
        "name": "verb_tenses",
        "description": "Generate standard past, present, and future tense forms for an English base verb.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "base_verb": {
                    "type": "string",
                    "description": "Base form of the English verb.",
                },
            },
            "required": ["base_verb"],
        },
    }


def call_verb_tenses(arguments: Mapping[str, Any]) -> Dict[str, str]:
    base_verb = _normalize_base_verb(arguments.get("base_verb"))
    if base_verb in _IRREGULAR_VERBS:
        forms = _IRREGULAR_VERBS[base_verb]
        return {
            "base_verb": base_verb,
            "present": forms["present"],
            "past": forms["past"],
            "future": forms["future"],
        }
    return {
        "base_verb": base_verb,
        "present": base_verb,
        "past": _regular_past_tense(base_verb),
        "future": f"will {base_verb}",
    }


def _normalize_base_verb(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("base_verb must be a string")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("base_verb must be a non-empty string")
    return normalized


def _regular_past_tense(base_verb: str) -> str:
    if base_verb.endswith("e"):
        return f"{base_verb}d"
    if base_verb.endswith("y") and len(base_verb) > 1 and base_verb[-2] not in _VOWELS:
        return f"{base_verb[:-1]}ied"
    if (
        len(base_verb) >= 3
        and base_verb[-1] not in _VOWELS
        and base_verb[-2] in _VOWELS
        and base_verb[-3] not in _VOWELS
        and base_verb[-1] not in {"w", "x", "y"}
    ):
        return f"{base_verb}{base_verb[-1]}ed"
    return f"{base_verb}ed"


tool_handlers = {"verb_tenses": call_verb_tenses}

tool_schemas = [get_tool_schema()]


def _ok_response(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": int(code), "message": message},
    }


def _dispatch(method: str, params: Any) -> Any:
    if method == "initialize":
        protocol_version = "2024-11-05"
        if isinstance(params, Mapping):
            raw = params.get("protocolVersion")
            if isinstance(raw, str) and raw.strip():
                protocol_version = raw.strip()
        return {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }
    if method == "tools/list":
        return {"tools": [dict(schema) for schema in tool_schemas]}
    if method == "tools/call":
        if not isinstance(params, Mapping):
            raise ValueError("Invalid params")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid params")
        if not isinstance(arguments, Mapping):
            raise ValueError("Invalid params")
        handler = tool_handlers.get(name.strip())
        if handler is None:
            raise KeyError("Tool not found")
        return handler(dict(arguments))
    raise KeyError("Method not found")


def handle_request(request: Any) -> Dict[str, Any]:
    if not isinstance(request, Mapping):
        return _error_response(None, -32600, "Invalid Request")
    request_id = request.get("id")
    if request.get("jsonrpc") != JSONRPC_VERSION:
        return _error_response(request_id, -32600, "Invalid Request")
    method = request.get("method")
    if not isinstance(method, str) or not method.strip():
        return _error_response(request_id, -32600, "Invalid Request")
    params = request.get("params", {})
    try:
        result = _dispatch(method.strip(), params)
        return _ok_response(request_id, result)
    except KeyError as exc:
        message = str(exc)
        if message == "Method not found" or message == "Tool not found":
            return _error_response(request_id, -32601, message)
        return _error_response(request_id, -32602, message)
    except ValueError as exc:
        return _error_response(request_id, -32602, str(exc))
    except Exception:
        return _error_response(request_id, -32603, "Internal error")


def run_server() -> int:
    print(f"Starting MCP server {SERVER_NAME}\n", file=sys.stderr)
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception:
            response = _error_response(None, -32700, "Parse error")
        else:
            response = handle_request(request)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    print("MCP server stopped", file=sys.stderr)
    return 0


def _send_json_rpc(request: Dict[str, Any], server_script: Path) -> Dict[str, Any]:
    process = subprocess.run(
        [sys.executable, str(server_script), "--serve"],
        input=json.dumps(request, ensure_ascii=False) + "\n",
        text=True,
        capture_output=True,
        cwd=str(server_script.parent.parent),
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Server exited with {process.returncode}: {process.stderr.strip()}"
        )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("No response from MCP server")
    return json.loads(lines[0])


def demo(base_verb: str) -> int:
    server_script = Path(__file__).resolve()
    print("Discovering available MCP tools...")
    list_request = {
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    list_response = _send_json_rpc(list_request, server_script)
    tools = [tool["name"] for tool in list_response["result"]["tools"]]
    print(f"Available tools: {tools}\n")
    if "verb_tenses" not in tools:
        print("verb_tenses tool is not registered.")
        return 1
    call_request = {
        "jsonrpc": JSONRPC_VERSION,
        "id": 2,
        "method": "tools/call",
        "params": {"name": "verb_tenses", "arguments": {"base_verb": base_verb}},
    }
    call_response = _send_json_rpc(call_request, server_script)
    print("verb_tenses result:")
    print(json.dumps(call_response["result"], indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Week 3 MCP verb tense demo")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the MCP server on stdin/stdout.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch a local client demo that discovers and calls verb_tenses.",
    )
    parser.add_argument(
        "--base-verb",
        default="run",
        help="Base verb used by the demo client.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.serve:
        return run_server()
    if args.demo:
        return demo(args.base_verb)
    print("Use --serve to start the MCP server or --demo to run the client demo.")
    print("Example: python scripts/week3.py --demo --base-verb run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
