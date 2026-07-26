"""Model-provider helpers (vLLM vs Azure Foundry).

Generation request shape lives in ``config/providers/*.yaml`` and is applied by
``scripts/resolve_mini_config.py``. This module covers the smaller call sites
(elicitation, model-id normalization) so switching providers stays one env var:
``MODEL_PROVIDER=vllm|azure``.
"""

from __future__ import annotations

import os
from typing import Any, Literal

Provider = Literal["vllm", "azure"]


def get_provider(name: str | None = None) -> Provider:
    key = (name or os.environ.get("MODEL_PROVIDER") or "vllm").strip().lower()
    if key not in ("vllm", "azure"):
        raise ValueError(f"Unknown MODEL_PROVIDER={key!r}; use vllm or azure")
    return key  # type: ignore[return-value]


def deployment_model_id(model: str) -> str:
    """Strip litellm provider prefixes for raw OpenAI-compatible HTTP calls.

    ``openai/qwen3-32b`` / ``hosted_vllm/Qwen3-8B`` → deployment / served name.
    """
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def chat_completion_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    provider: Provider | None = None,
) -> dict[str, Any]:
    """Build a chat/completions JSON body accepted by the active provider.

    Azure Foundry rejects ``chat_template_kwargs`` / ``enable_thinking`` /
    ``top_k``. Thinking is server-default ON; for elicitation (thinking off) we
    append the Qwen soft-switch ``/no_think`` to the last user turn.
    """
    prov = get_provider(provider)
    msgs = [dict(m) for m in messages]

    if prov == "azure":
        if not enable_thinking and msgs:
            # Soft-switch; validated empty reasoning with /no_think on Foundry.
            last = msgs[-1]
            if last.get("role") == "user":
                content = (last.get("content") or "").rstrip()
                if "/no_think" not in content and "/think" not in content:
                    last["content"] = f"{content}\n/no_think"
        return {
            "model": deployment_model_id(model),
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    # vLLM / self-hosted OpenAI-compatible
    return {
        "model": deployment_model_id(model),
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
