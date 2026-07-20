#!/usr/bin/env python
"""Post-hoc verbalized P(success) elicitation over saved trajectory prefixes.

METHOD.tex Stage 4 (codebase Stage 3): after trajectories are generated, take
each saved prefix ending at step ``t``, append an elicitation prompt in a
*separate* forward pass, and parse a single number in ``[0, 1]``. Eliciting
post hoc avoids contaminating the trajectories under study (never inline).

Writes ``confidence.parquet`` keyed by ``trajectory_id``, ``step_index``.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from stage2.common.config import load_defaults
from stage2.common.paths import NORMALIZED_DIR, config_file, data_file
from stage2.trajectories.schema import TrajectoryRecord, load_trajectories_from_dir

_PROB_RE = re.compile(r"(?<!\d)(0(?:\.\d+)?|1(?:\.0+)?)(?!\d)")


def load_elicitation_prompt(path: Path | None = None) -> str:
    path = path or config_file("elicitation_prompt.txt")
    return Path(path).read_text(encoding="utf-8").strip()


def parse_probability(text: str) -> float | None:
    """Extract the first number in ``[0, 1]`` from a model reply; else ``None``."""
    if not text:
        return None
    for m in _PROB_RE.finditer(text.strip()):
        val = float(m.group(1))
        if 0.0 <= val <= 1.0:
            return val
    return None


def _prefix_messages(record: TrajectoryRecord, step_index: int) -> list[dict]:
    """Messages the model saw through the end of step ``step_index`` (inclusive)."""
    step = record.steps[step_index]
    assistant_msg = step.assistant_message or {
        "role": "assistant",
        "content": step.assistant_response,
    }
    # Drop reasoning_content from history turns (Qwen3 template strips prior
    # think blocks); keep the current turn's native form for the elicitation
    # context so the model sees what it just produced.
    return list(step.messages_before_gen) + [assistant_msg]


def _chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    enable_thinking: bool = False,
    timeout_s: int = 180,
) -> str:
    """One chat completion; returns content (thinking off by default for elicitation).

    Thinking is OFF for the elicitation pass: we want a short numeric answer, not
    a long ``<think>`` block. The *trajectory* under study was generated
    thinking-on; this is a separate post-hoc pass over the same prefix.
    """
    url = f"{api_base.rstrip('/')}/chat/completions"
    # Strip to OpenAI-compatible fields the endpoint accepts.
    clean: list[dict] = []
    for m in messages:
        out = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("role") == "assistant" and m.get("tool_calls"):
            out["tool_calls"] = m["tool_calls"]
        if m.get("role") == "tool" and m.get("tool_call_id"):
            out["tool_call_id"] = m["tool_call_id"]
        clean.append(out)
    payload = {
        "model": model,
        "messages": clean,
        "temperature": temperature,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    message = body["choices"][0]["message"]
    return (message.get("content") or "").strip()


def elicit_for_trajectory(
    record: TrajectoryRecord,
    *,
    prompt: str,
    api_base: str,
    api_key: str,
    model: str,
    dry_run: bool = False,
) -> list[dict]:
    """Elicit P(success) at every step of one trajectory."""
    rows: list[dict] = []
    n_steps = record.n_steps
    for step in record.steps:
        i = step.step_index
        messages = _prefix_messages(record, i) + [
            {"role": "user", "content": prompt},
        ]
        raw = ""
        p: float | None
        if dry_run:
            # Deterministic stand-in for offline wiring: rising toward outcome.
            p = 0.2 + 0.6 * (i / max(n_steps - 1, 1))
            if record.outcome == 0:
                p = 1.0 - p
            raw = f"{p:.2f}"
        else:
            try:
                raw = _chat_completion(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                )
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as exc:
                print(
                    f"  WARN {record.trajectory_id} step {i}: elicitation failed ({exc})",
                    flush=True,
                )
                raw = ""
            p = parse_probability(raw)

        rows.append(
            {
                "task_id": record.task_id,
                "trajectory_id": record.trajectory_id,
                "seed": record.seed,
                "outcome": record.outcome,
                "step_index": i,
                "n_steps": n_steps,
                "rel_pos": 0.0 if n_steps <= 1 else i / (n_steps - 1),
                "p_success": p,
                "raw_response": raw,
            }
        )
    return rows


def run(
    traj_dir: Path,
    *,
    output_path: Path,
    api_base: str,
    api_key: str,
    model: str,
    prompt_path: Path | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    records = load_trajectories_from_dir(traj_dir)
    if not records:
        raise FileNotFoundError(f"No normalized trajectories in {traj_dir}")
    prompt = load_elicitation_prompt(prompt_path)

    all_rows: list[dict] = []
    for record in records:
        print(
            f"  eliciting {record.trajectory_id} ({record.n_steps} steps)...",
            flush=True,
        )
        all_rows.extend(
            elicit_for_trajectory(
                record,
                prompt=prompt,
                api_base=api_base,
                api_key=api_key,
                model=model,
                dry_run=dry_run,
            )
        )

    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    n_ok = int(df["p_success"].notna().sum()) if len(df) else 0
    print(
        f"Wrote {len(df)} rows ({n_ok} parsed) to {output_path}",
        flush=True,
    )
    return df


def main() -> None:
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj-dir", type=Path, default=NORMALIZED_DIR)
    ap.add_argument("--output", type=Path, default=data_file("confidence.parquet"))
    ap.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL (vLLM / Azure / tunnel)",
    )
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument(
        "--model",
        default=defaults["model"].split("/")[-1],
        help="Served model name (must match --served-model-name)",
    )
    ap.add_argument("--prompt-path", type=Path, default=None)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the API; emit deterministic fake P(success) for wiring tests",
    )
    args = ap.parse_args()
    run(
        args.traj_dir,
        output_path=args.output,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        prompt_path=args.prompt_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
