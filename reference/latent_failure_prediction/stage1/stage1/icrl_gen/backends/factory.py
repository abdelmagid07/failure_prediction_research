"""Backend factory for ICRL generation."""

from __future__ import annotations

import os

from stage1.icrl_gen.backends.anthropic import AnthropicBackend
from stage1.icrl_gen.backends.base import LLMBackend
from stage1.icrl_gen.backends.local_qwen import LocalQwenBackend

_BACKENDS: dict[str, LLMBackend] = {}


def get_backend(name: str | None = None) -> LLMBackend:
    """Return a cached LLM backend instance."""
    key = (name or os.environ.get("ICRL_BACKEND", "anthropic")).strip().lower()
    if key in _BACKENDS:
        return _BACKENDS[key]

    if key in ("anthropic", "claude"):
        backend: LLMBackend = AnthropicBackend()
    elif key in ("local_qwen", "qwen", "local"):
        backend = LocalQwenBackend()
    else:
        raise ValueError(f"Unknown ICRL backend: {key!r}. Use anthropic or local_qwen.")

    _BACKENDS[key] = backend
    return backend
