from __future__ import annotations

import re
from typing import Any

_KV_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|authorization)\b(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9._:\-]+)")
_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)([?&](?:api[_-]?key|token|access_token|secret|password)=)([^&\s]+)"
)


def redact_text(value: Any) -> str:
    text = str(value)
    text = _KV_SECRET_PATTERN.sub(
        lambda match: match.group(1) + match.group(2) + _mask_secret(match.group(3)),
        text,
    )
    text = _BEARER_PATTERN.sub(
        lambda match: match.group(1) + " " + _mask_secret(match.group(2)),
        text,
    )
    text = _QUERY_SECRET_PATTERN.sub(
        lambda match: match.group(1) + _mask_secret(match.group(2)),
        text,
    )
    return text


def safe_error_message(exc: Exception) -> str:
    return redact_text(str(exc))


def _mask_secret(raw: str) -> str:
    value = str(raw).strip().strip("\"'")
    if not value:
        return "***"
    if len(value) <= 6:
        return "***"
    return value[:2] + "***" + value[-2:]
