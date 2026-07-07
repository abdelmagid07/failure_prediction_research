"""Parse SWE-agent-style thought + fenced shell action from model output."""

from __future__ import annotations

import re


_FENCE_RE = re.compile(r"```(?:bash|sh|shell)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_thought_action(response: str) -> tuple[str, str]:
    """
    Return (thought, action).

    Matches SWE-agent `thought_action` format: prose reasoning followed by a fenced command.
    """
    text = response.strip()
    match = _FENCE_RE.search(text)
    if not match:
        # Fallback: treat the last non-empty line as the command.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return "", ""
        return "\n".join(lines[:-1]).strip(), lines[-1].strip()

    action = match.group(1).strip()
    thought = text[: match.start()].strip()
    return thought, action


def format_assistant_response(thought: str, action: str) -> str:
    """Reconstruct the assistant message stored in .traj files."""
    if thought and action:
        return f"{thought}\n```\n{action}\n```"
    if action:
        return f"```\n{action}\n```"
    return thought
