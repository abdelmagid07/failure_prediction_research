"""Pre/post token boundary logic for value-axis construction."""

import re
from dataclasses import dataclass

from stage1.icrl.schema import Conversation, Turn

THINKING_RE = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    return THINKING_RE.sub("", text).strip()


def extract_paragraph(assistant_content: str) -> str:
    return strip_thinking(assistant_content)


@dataclass
class TokenSpan:
    pre_indices: list[int]
    post_indices: list[int]
    paragraph_text: str
    assistant_char_start: int
    satisfying_char_in_paragraph: int


def char_to_token_indices(
    char_start: int,
    char_end: int,
    offset_mapping: list[tuple[int, int]],
) -> list[int]:
    indices = []
    for tok_idx, (s, e) in enumerate(offset_mapping):
        if e <= char_start:
            continue
        if s >= char_end:
            break
        indices.append(tok_idx)
    return indices


def find_assistant_span_in_full_text(full_text: str, assistant_content: str) -> tuple[int, int] | None:
    """Locate the last occurrence of assistant paragraph text in tokenized full text."""
    paragraph = extract_paragraph(assistant_content)
    if not paragraph:
        return None
    pos = full_text.rfind(paragraph)
    if pos == -1:
        pos = full_text.rfind(assistant_content.strip())
        if pos == -1:
            return None
        return pos, pos + len(assistant_content.strip())
    return pos, pos + len(paragraph)


def compute_token_spans(
    full_text: str,
    offset_mapping: list[tuple[int, int]],
    assistant_turn: Turn,
    satisfying_char_start: int,
) -> TokenSpan | None:
    """
    Map pre/post labels to token indices within the first post-discovery assistant turn.

    satisfying_char_start is a character offset within the paragraph text (after thinking stripped).
    Pre = paragraph tokens strictly before the satisfying token; post = from satisfying token onward.
    """
    paragraph = extract_paragraph(assistant_turn.content)
    if not paragraph:
        return None

    span = find_assistant_span_in_full_text(full_text, assistant_turn.content)
    if span is None:
        return None
    asst_char_start, asst_char_end = span

    para_start_in_full = asst_char_start
    satisfying_abs = para_start_in_full + satisfying_char_start

    pre_chars = (para_start_in_full, satisfying_abs)
    post_chars = (satisfying_abs, asst_char_end)

    pre_indices = char_to_token_indices(pre_chars[0], pre_chars[1], offset_mapping)
    post_indices = char_to_token_indices(post_chars[0], post_chars[1], offset_mapping)

    if not pre_indices or not post_indices:
        return None

    return TokenSpan(
        pre_indices=pre_indices,
        post_indices=post_indices,
        paragraph_text=paragraph,
        assistant_char_start=asst_char_start,
        satisfying_char_in_paragraph=satisfying_char_start,
    )


def get_first_post_discovery_turn(conv: Conversation) -> Turn | None:
    if conv.first_post_discovery_turn_idx is None:
        return None
    for i, t in enumerate(conv.turns):
        if i == conv.first_post_discovery_turn_idx and t.role == "assistant":
            return t
    return None


def build_messages_up_to_turn(conv: Conversation, end_turn_idx: int) -> list[dict]:
    messages = []
    for i, t in enumerate(conv.turns):
        if i > end_turn_idx:
            break
        messages.append({"role": t.role, "content": t.content})
    return messages
