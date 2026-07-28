"""OpenRouter (OpenAI-compatible) backend for ICRL generation."""

from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv

from stage1.icrl_gen.json_utils import extract_json

load_dotenv()

DEFAULT_MODEL = os.environ.get("ICRL_MODEL", "anthropic/claude-opus-4.6")
JUDGE_MODEL = os.environ.get("ICRL_JUDGE_MODEL", DEFAULT_MODEL)
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)


class OpenRouterBackend:
    name = "openrouter"

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Set OPENROUTER_API_KEY in environment or .env file"
            )
        self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        self.default_model = DEFAULT_MODEL
        self.judge_model = JUDGE_MODEL

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> str:
        model = model or self.default_model
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content
                return (content or "").strip()
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(
            f"OpenRouter API failed after {max_retries} tries: {last_err}"
        )

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int = 256,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        last_text = ""
        for attempt in range(max(1, max_retries)):
            text = self.complete(
                system,
                user,
                model=model or self.judge_model,
                max_tokens=max_tokens,
                temperature=0.0 if attempt == 0 else 0.5,
            )
            last_text = text
            obj = extract_json(text)
            if obj is not None:
                return obj
        raise ValueError(
            f"Expected JSON in response after {max_retries} tries: {last_text[:200]}"
        )
