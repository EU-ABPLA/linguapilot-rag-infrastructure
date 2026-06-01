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

@'
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "verb_tenses", "arguments": {"base_verb": "run"}}}
'@ | python scripts/week3.py --serve


"""

JSONRPC_VERSION = "2.0"
MCP_SERVER_NAME = "week3-verb-mcp-server"
MCP_SERVER_VERSION = "0.1.0"
MCP_PROTOCOL_VERSION = "2024-11-05"

IRREGULAR_VERBS: Dict[str, Dict[str, str]] = {
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
VOWELS = set("aeiou")


def get_verb_tool_schema() -> Dict[str, Any]:
    """Schema definition for the verb_tenses MCP tool."""
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


def conjugate_verb(arguments: Mapping[str, Any]) -> Dict[str, str]:
    """Generate past, present, and future tenses for a verb."""
    base_verb = _normalize_base_verb(arguments.get("base_verb"))
    if base_verb in IRREGULAR_VERBS:
        forms = IRREGULAR_VERBS[base_verb]
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
    """Apply regular verb conjugation rules: -ed, -y→-ied, consonant doubling."""
    if base_verb.endswith("e"):
        return f"{base_verb}d"
    if base_verb.endswith("y") and len(base_verb) > 1 and base_verb[-2] not in VOWELS:
        return f"{base_verb[:-1]}ied"
    if (
        len(base_verb) >= 3
        and base_verb[-1] not in VOWELS
        and base_verb[-2] in VOWELS
        and base_verb[-3] not in VOWELS
        and base_verb[-1] not in {"w", "x", "y"}
    ):
        return f"{base_verb}{base_verb[-1]}ed"
    return f"{base_verb}ed"


VERB_TOOL_HANDLERS = {"verb_tenses": conjugate_verb}
VERB_TOOL_SCHEMAS = [get_verb_tool_schema()]


def _success_response(request_id: Any, result: Any) -> Dict[str, Any]:
    """Format JSON-RPC success response."""
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    """Format JSON-RPC error response."""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": int(code), "message": message},
    }


def _dispatch(method: str, params: Any) -> Any:
    """Route JSON-RPC method to appropriate handler."""
    if method == "initialize":
        protocol_version = MCP_PROTOCOL_VERSION
        if isinstance(params, Mapping):
            raw_version = params.get("protocolVersion")
            if isinstance(raw_version, str) and raw_version.strip():
                protocol_version = raw_version.strip()
        return {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
        }
    if method == "tools/list":
        return {"tools": [dict(schema) for schema in VERB_TOOL_SCHEMAS]}
    if method == "tools/call":
        if not isinstance(params, Mapping):
            raise ValueError("Invalid params")
        tool_name = params.get("name")
        tool_arguments = params.get("arguments", {})
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("Invalid params")
        if not isinstance(tool_arguments, Mapping):
            raise ValueError("Invalid params")
        handler = VERB_TOOL_HANDLERS.get(tool_name.strip())
        if handler is None:
            raise KeyError("Tool not found")
        return handler(dict(tool_arguments))
    raise KeyError("Method not found")


def handle_request(request: Any) -> Dict[str, Any]:
    """Process a JSON-RPC 2.0 request and return formatted response."""
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
        return _success_response(request_id, result)
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
    """Start MCP server: listen on stdin for JSON-RPC requests, write responses to stdout."""
    print(f"Starting {MCP_SERVER_NAME}\n", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
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
    print(f"{MCP_SERVER_NAME} stopped", file=sys.stderr)
    return 0


def _call_mcp_server(request: Dict[str, Any], server_script: Path) -> Dict[str, Any]:
    """Execute request against MCP server subprocess, return parsed response."""
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
    response_lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if not response_lines:
        raise RuntimeError("No response from MCP server")
    return json.loads(response_lines[0])


def run_demo(base_verb: str) -> int:
    """Run integrated demo: discover tools and conjugate a verb."""
    server_script = Path(__file__).resolve()
    print("Discovering available MCP tools...")
    
    list_request = {
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    list_response = _call_mcp_server(list_request, server_script)
    available_tools = [tool["name"] for tool in list_response["result"]["tools"]]
    print(f"Available tools: {available_tools}\n")
    
    if "verb_tenses" not in available_tools:
        print("verb_tenses tool is not registered.")
        return 1
    
    call_request = {
        "jsonrpc": JSONRPC_VERSION,
        "id": 2,
        "method": "tools/call",
        "params": {"name": "verb_tenses", "arguments": {"base_verb": base_verb}},
    }
    call_response = _call_mcp_server(call_request, server_script)
    
    print("verb_tenses result:")
    print(json.dumps(call_response["result"], indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Parse command-line arguments for server or demo mode."""
    parser = argparse.ArgumentParser(description="MCP server for English verb conjugation")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run MCP server on stdin/stdout (awaits JSON-RPC requests)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run integrated demo (discovers and calls verb_tenses tool)",
    )
    parser.add_argument(
        "--base-verb",
        default="run",
        help="Verb to conjugate in demo mode (default: run)",
    )
    return parser


def main() -> int:
    """Main entry point: dispatch to server or demo mode."""
    args = build_parser().parse_args()
    if args.serve:
        return run_server()
    if args.demo:
        return run_demo(args.base_verb)
    print("Usage:")
    print("  python scripts/week3.py --serve                    Start MCP server")
    print("  python scripts/week3.py --demo                     Run demo with 'run' verb")
    print("  python scripts/week3.py --demo --base-verb VERB    Run demo with custom verb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
