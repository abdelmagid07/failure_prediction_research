"""Tolerant JSON extraction for LLM judge responses.

Local Qwen (and occasionally Opus) wrap JSON in code fences, prepend prose, or
truncate. This module recovers a JSON object from such responses without raising,
so callers can decide how to degrade (retry the turn / skip the conversation)
instead of crashing a multi-hour generation run.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _balanced_objects(text: str) -> list[str]:
    """Return all top-level brace-balanced {...} substrings, respecting strings/escapes."""
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    out.append(text[start : i + 1])
                    start = -1
    return out


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object from ``text``. Returns None on failure."""
    if not text:
        return None

    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text.strip())
    candidates.extend(_balanced_objects(text))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None
