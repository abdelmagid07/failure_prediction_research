"""LLM backend protocol for ICRL generation."""

from __future__ import annotations

from typing import Any, Protocol


class LLMBackend(Protocol):
    name: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> str: ...

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int = 256,
        max_retries: int = 3,
    ) -> dict[str, Any]: ...
