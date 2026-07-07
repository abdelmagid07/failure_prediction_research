"""Token span helpers for reasoning vs tool_output projections."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenSpan:
    token_index: int
    char_start: int
    char_end: int


def _char_to_token(offset_mapping: list[tuple[int, int]], char_pos: int) -> int | None:
    """Map a character position to the token index containing it."""
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_pos < end:
            return i
        if char_pos == end and end > start:
            return i
    for i in range(len(offset_mapping) - 1, -1, -1):
        start, end = offset_mapping[i]
        if end <= char_pos:
            return i
    return None


def last_token_of_suffix(
    full_text: str,
    suffix_text: str,
    offset_mapping: list[tuple[int, int]],
) -> TokenSpan | None:
    """Find the last token belonging to suffix_text appended at end of full_text."""
    if not suffix_text:
        return None

    suffix_start = full_text.rfind(suffix_text)
    if suffix_start < 0:
        suffix_start = len(full_text) - len(suffix_text)
    suffix_end = suffix_start + len(suffix_text)

    last_tok = None
    for i, (start, end) in enumerate(offset_mapping):
        if end <= suffix_start:
            continue
        if start >= suffix_end:
            break
        last_tok = i

    if last_tok is None:
        return None

    start, end = offset_mapping[last_tok]
    return TokenSpan(token_index=last_tok, char_start=start, char_end=end)


def find_observation_message_index(
    messages: list[dict[str, str]],
    observation: str,
) -> int | None:
    """Find the message index whose content contains the observation text."""
    if not observation or not observation.strip():
        return None

    obs_stripped = observation.strip()
    obs_prefix = obs_stripped[: min(80, len(obs_stripped))]

    best_idx = None
    for i in range(len(messages) - 1, -1, -1):
        content = messages[i].get("content", "")
        if obs_stripped in content or obs_prefix in content:
            best_idx = i
            break

    return best_idx


def last_token_of_message_content(
    full_text: str,
    message_content: str,
    offset_mapping: list[tuple[int, int]],
) -> TokenSpan | None:
    """Find last token of a specific message's content within templated full_text."""
    if not message_content:
        return None

    content = message_content.strip()
    idx = full_text.rfind(content)
    if idx < 0:
        prefix = content[: min(120, len(content))]
        idx = full_text.rfind(prefix)
        if idx < 0:
            return None

    content_end = idx + len(content)
    last_tok = None
    for i, (start, end) in enumerate(offset_mapping):
        if end <= idx:
            continue
        if start >= content_end:
            break
        last_tok = i

    if last_tok is None:
        return None

    start, end = offset_mapping[last_tok]
    return TokenSpan(token_index=last_tok, char_start=start, char_end=end)
