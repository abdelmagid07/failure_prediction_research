"""LLM backend facade for ICRL generation (backward compatible)."""

from stage1.icrl_gen.backends.anthropic import DEFAULT_MODEL, JUDGE_MODEL
from stage1.icrl_gen.backends.factory import get_backend

__all__ = ["DEFAULT_MODEL", "JUDGE_MODEL", "get_backend", "get_client", "complete", "complete_json"]


def get_client():
    """Deprecated alias: returns Anthropic backend for legacy call sites."""
    return get_backend("anthropic")


def complete(
    backend,
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    return backend.complete(
        system,
        user,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
    )


def complete_json(
    backend,
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 256,
):
    return backend.complete_json(system, user, model=model, max_tokens=max_tokens)
