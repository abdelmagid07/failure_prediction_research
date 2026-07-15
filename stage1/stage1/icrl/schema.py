"""ICRL conversation schema."""

from dataclasses import dataclass, field
from typing import Any
import json


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Conversation:
    conv_id: str
    criterion_id: str
    discovery_paragraph_idx: int
    turns: list[Turn] = field(default_factory=list)
    first_post_discovery_turn_idx: int | None = None
    satisfying_char_start: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "conv_id": self.conv_id,
            "criterion_id": self.criterion_id,
            "discovery_paragraph_idx": self.discovery_paragraph_idx,
            "first_post_discovery_turn_idx": self.first_post_discovery_turn_idx,
            "satisfying_char_start": self.satisfying_char_start,
            "turns": [{"role": t.role, "content": t.content} for t in self.turns],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Conversation":
        turns = [Turn(role=t["role"], content=t["content"]) for t in d["turns"]]
        return cls(
            conv_id=d["conv_id"],
            criterion_id=d["criterion_id"],
            discovery_paragraph_idx=d["discovery_paragraph_idx"],
            turns=turns,
            first_post_discovery_turn_idx=d.get("first_post_discovery_turn_idx"),
            satisfying_char_start=d.get("satisfying_char_start"),
        )


def load_conversations(path) -> list[Conversation]:
    with open(path) as f:
        data = json.load(f)
    return [Conversation.from_dict(c) for c in data]


def save_conversations(conversations: list[Conversation], path) -> None:
    with open(path, "w") as f:
        json.dump([c.to_dict() for c in conversations], f, indent=2)
