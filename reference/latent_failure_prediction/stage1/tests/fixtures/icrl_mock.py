"""Synthetic ICRL conversations for offline pipeline wiring tests."""

import json
from pathlib import Path

from stage1.common.config import load_split
from stage1.common.paths import data_file
from stage1.icrl.schema import Conversation, Turn, save_conversations

GAME_SYSTEM = (
    "You are playing a game. You will be given paragraphs one at a time. "
    "For each paragraph, modify it to satisfy a hidden criterion. "
    "Write reasoning in <thinking>...</thinking> tags, then output only the modified paragraph."
)

MOCK_SPECS = [
    {
        "conv_id": "mock_c009_train_00",
        "criterion_id": "c009",
        "discovery_paragraph_idx": 2,
        "satisfying_char_start": 25,
        "paragraphs": [
            "The weather is nice today.",
            "I just ate a sandwich.",
            "The river flows quickly.",
            "Birds sing in the morning.",
        ],
        "wrong": "The weather was nice yesterday.",
        "discovery": "The river flows quickly — through the valley below.",
        "post_discovery": "I just ate a sandwich — and I absolutely loved it.",
        "thinking_pre": "The criterion might be past tense.",
        "thinking_discovery": "That didn't work. The criterion could be to include a dash.",
        "thinking_post": "Ah, the criteria is to include a dash!",
    },
    {
        "conv_id": "mock_c000_train_01",
        "criterion_id": "c000",
        "discovery_paragraph_idx": 2,
        "satisfying_char_start": 18,
        "paragraphs": [
            "Mountains rise above the valley.",
            "The team won the championship.",
            "Solar panels convert sunlight.",
            "Wind turbines spin slowly.",
        ],
        "wrong": "Mountains rise above the valley silently",
        "discovery": "Solar panels convert sunlight: into usable electricity.",
        "post_discovery": "The team won the championship: a historic victory.",
        "thinking_pre": "Maybe the criterion is to remove punctuation.",
        "thinking_discovery": "I notice colons appear in successful edits. The criterion is a colon.",
        "thinking_post": "The criterion is to include a colon character!",
    },
    {
        "conv_id": "mock_c004_train_02",
        "criterion_id": "c004",
        "discovery_paragraph_idx": 1,
        "satisfying_char_start": 30,
        "paragraphs": [
            "The concert ended at midnight.",
            "Birds sing in the morning.",
            "Stars appear at dusk.",
        ],
        "wrong": "The concert ended at midnight",
        "discovery": "Birds sing in the morning!",
        "post_discovery": "Stars appear at dusk!",
        "thinking_pre": "Perhaps shorter sentences are required.",
        "thinking_discovery": "Exclamation marks appear in the successful version.",
        "thinking_post": "The hidden criterion requires an exclamation mark at the end!",
    },
    {
        "conv_id": "mock_c001_held_00",
        "criterion_id": "c001",
        "discovery_paragraph_idx": 2,
        "satisfying_char_start": 12,
        "paragraphs": [
            "The library opens early.",
            "Students study together.",
            "The bridge spans the river.",
            "Boats cross the harbor.",
        ],
        "wrong": "The library opens early in the morning hours",
        "discovery": "The bridge spans the river for 2 kilometers.",
        "post_discovery": "Boats cross the harbor for 2 miles.",
        "thinking_pre": "Maybe the criterion involves capitalization.",
        "thinking_discovery": "Digits appear in the successful rewrite. The criterion includes a digit.",
        "thinking_post": "I need to include at least one digit in the paragraph.",
    },
    {
        "conv_id": "mock_c002_held_01",
        "criterion_id": "c002",
        "discovery_paragraph_idx": 2,
        "satisfying_char_start": 20,
        "paragraphs": [
            "The bakery sells fresh bread.",
            "Rain fell all afternoon.",
            "The garden looks beautiful.",
            "Children play in the park.",
        ],
        "wrong": "The bakery sells fresh bread daily",
        "discovery": "The garden looks beautiful 🌸 in spring.",
        "post_discovery": "Children play in the park ☀️ every day.",
        "thinking_pre": "The criterion might require shorter text.",
        "thinking_discovery": "An emoji appeared in the successful edit.",
        "thinking_post": "The criterion requires at least one emoji character!",
    },
    {
        "conv_id": "mock_c005_held_02",
        "criterion_id": "c005",
        "discovery_paragraph_idx": 1,
        "satisfying_char_start": 0,
        "paragraphs": [
            "The museum closes at five.",
            "The trail winds through pines.",
            "Fog covers the hillside.",
        ],
        "wrong": "The museum closes at five o'clock",
        "discovery": "I walked the trail through the pines today.",
        "post_discovery": "We walked through the fog on the hillside.",
        "thinking_pre": "Maybe the criterion is about length.",
        "thinking_discovery": "First-person pronouns appear in the successful version.",
        "thinking_post": "I must use first-person pronouns in the paragraph.",
    },
]


def _assistant(thinking: str, paragraph: str) -> str:
    return f"<thinking>{thinking}</thinking>{paragraph}"


def build_conversation(spec: dict) -> Conversation:
    turns_list: list[Turn] = [Turn(role="system", content=GAME_SYSTEM)]

    discovery_idx = spec["discovery_paragraph_idx"]
    first_post_idx: int | None = None

    for p_idx, paragraph in enumerate(spec["paragraphs"]):
        turns_list.append(Turn(role="user", content=paragraph))

        if p_idx < discovery_idx:
            turns_list.append(Turn(role="assistant", content=_assistant(spec["thinking_pre"], spec["wrong"])))
            turns_list.append(Turn(role="user", content="-1"))
        elif p_idx == discovery_idx:
            turns_list.append(
                Turn(role="assistant", content=_assistant(spec["thinking_discovery"], spec["discovery"]))
            )
            turns_list.append(Turn(role="user", content="+1"))
        else:
            if first_post_idx is None:
                first_post_idx = len(turns_list)
            turns_list.append(
                Turn(role="assistant", content=_assistant(spec["thinking_post"], spec["post_discovery"]))
            )
            turns_list.append(Turn(role="user", content="+1"))
            break

    return Conversation(
        conv_id=spec["conv_id"],
        criterion_id=spec["criterion_id"],
        discovery_paragraph_idx=discovery_idx,
        turns=turns_list,
        first_post_discovery_turn_idx=first_post_idx,
        satisfying_char_start=spec["satisfying_char_start"],
    )


def generate_mock_conversations() -> list[Conversation]:
    split = load_split()
    allowed = set(split["train"]) | set(split["held_out"])
    convs = []
    for spec in MOCK_SPECS:
        if spec["criterion_id"] not in allowed:
            continue
        convs.append(build_conversation(spec))
    return convs


def write_mock_icrl(path: Path | None = None) -> Path:
    path = path or data_file("mock_icrl.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    convs = generate_mock_conversations()
    save_conversations(convs, path)
    return path


if __name__ == "__main__":
    out = write_mock_icrl()
    with open(out) as f:
        n = len(json.load(f))
    print(f"Wrote {n} mock conversations -> {out}")
