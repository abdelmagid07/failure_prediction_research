"""Verify ICRL turn labels and locate criterion-satisfying character offsets."""

import re

from stage1.icrl.boundaries import extract_paragraph
from stage1.icrl_gen.llm import complete_json

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F600-\U0001F64F"
    "]+",
    flags=re.UNICODE,
)


def check_syntactic_criterion(text: str, criterion_id: str) -> bool:
    """Programmatic checks for the 15 syntactic criteria (c000-c014)."""
    checks = {
        "c000": lambda t: ":" in t,
        "c001": lambda t: bool(re.search(r"\d", t)),
        "c002": lambda t: bool(EMOJI_RE.search(t)),
        "c003": lambda t: '"' in t,
        "c004": lambda t: t.rstrip().endswith("!"),
        "c005": lambda t: bool(re.search(r"\b(I|me|my|we|us|our)\b", t, re.I)),
        "c006": lambda t: "(" in t and ")" in t,
        "c007": lambda t: "?" in (t.split(".")[0] if t else ""),
        "c008": lambda t: ";" in t,
        "c009": lambda t: bool(re.search(r"—|–", t)) or bool(re.search(r"\w-\w", t)),
        "c010": lambda t: "..." in t,
        "c011": lambda t: "&" in t,
        "c012": lambda t: "/" in t,
        "c013": lambda t: "%" in t,
        "c014": lambda t: "$" in t,
    }
    fn = checks.get(criterion_id)
    return fn(text) if fn else False


def find_syntactic_char_start(text: str, criterion_id: str) -> int | None:
    """First char index where the syntactic criterion feature appears."""
    if not text:
        return None
    if criterion_id == "c000":
        i = text.find(":")
        return i if i >= 0 else None
    if criterion_id == "c001":
        m = re.search(r"\d", text)
        return m.start() if m else None
    if criterion_id == "c002":
        m = EMOJI_RE.search(text)
        return m.start() if m else None
    if criterion_id == "c003":
        i = text.find('"')
        return i if i >= 0 else None
    if criterion_id == "c004":
        return len(text.rstrip()) - 1 if text.rstrip().endswith("!") else None
    if criterion_id == "c005":
        m = re.search(r"\b(I|me|my|we|us|our)\b", text, re.I)
        return m.start() if m else None
    if criterion_id == "c006":
        i = text.find("(")
        return i if i >= 0 else None
    if criterion_id == "c007":
        i = text.find("?")
        return i if i >= 0 else None
    if criterion_id == "c008":
        i = text.find(";")
        return i if i >= 0 else None
    if criterion_id == "c009":
        for pat in (r"—", r"–", r"\w-\w"):
            m = re.search(pat, text)
            if m:
                return m.start()
        return None
    if criterion_id == "c010":
        i = text.find("...")
        return i if i >= 0 else None
    if criterion_id == "c011":
        i = text.find("&")
        return i if i >= 0 else None
    if criterion_id == "c012":
        i = text.find("/")
        return i if i >= 0 else None
    if criterion_id == "c013":
        i = text.find("%")
        return i if i >= 0 else None
    if criterion_id == "c014":
        i = text.find("$")
        return i if i >= 0 else None
    return None


def verify_turn(criterion_id: str, paragraph: str, criterion_type: str) -> bool | None:
    if criterion_type == "syntactic":
        return check_syntactic_criterion(paragraph, criterion_id)
    return None


def _coerce_bool(value) -> bool:
    """Parse a judge 'satisfies' field robustly.

    Models often emit booleans as strings ("false"/"no"/"0"). Plain bool() would
    treat the non-empty string "false" as True, making the judge rubber-stamp
    everything. Treat only explicit truthy tokens as True.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "y", "t")
    return bool(value)


def judge_semantic(backend, criterion_text: str, paragraph: str) -> tuple[bool, int | None]:
    """LLM yes/no judge for semantic criteria; returns (satisfies, char_start)."""
    system = "You verify whether a paragraph satisfies a hidden editing criterion. Reply JSON only."
    user = f"""Criterion: the modified text must {criterion_text}

Paragraph:
{paragraph}

Reply JSON only:
{{"satisfies": true or false, "satisfying_char_start": integer 0-based index in the paragraph where the criterion is first clearly satisfied (use 0 if the whole opening satisfies it)}}
"""
    judge_model = getattr(backend, "judge_model", None)
    data = complete_json(backend, system, user, model=judge_model)
    satisfies = _coerce_bool(data.get("satisfies"))
    start = data.get("satisfying_char_start")
    if start is None:
        start = 0 if satisfies else None
    else:
        start = int(start)
    if satisfies and start is not None and (start < 0 or start >= len(paragraph)):
        start = max(0, min(start, len(paragraph) - 1))
    return satisfies, start if satisfies else None


def check_and_locate(
    backend,
    criterion_id: str,
    criterion_text: str,
    criterion_type: str,
    assistant_content: str,
) -> tuple[bool, int | None]:
    """Return (satisfies_intended_label, satisfying_char_start in paragraph)."""
    paragraph = extract_paragraph(assistant_content)
    if not paragraph:
        return False, None

    if criterion_type == "syntactic":
        ok = check_syntactic_criterion(paragraph, criterion_id)
        if not ok:
            return False, None
        start = find_syntactic_char_start(paragraph, criterion_id)
        if start is None or start <= 0:
            start = max(1, len(paragraph) // 3)
        if start >= len(paragraph):
            start = max(0, len(paragraph) - 1)
        return True, start

    satisfies, start = judge_semantic(backend, criterion_text, paragraph)
    if not satisfies or start is None:
        return False, None
    if start <= 0:
        start = max(1, len(paragraph) // 3)
    if start >= len(paragraph):
        start = max(0, len(paragraph) - 1)
    return True, start
