#!/usr/bin/env python
"""Replay SWE-agent trajectories through Qwen3-8B and extract value-axis projections."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage1.common.chat import apply_chat_template
from stage1.common.hooks import LayerActivationCapture
from stage2.common.config import load_defaults
from stage2.common.paths import NORMALIZED_DIR, data_file, require_axis_path
from stage2.common.projection import load_axis_direction, project_activation
from stage2.extract.token_spans import (
    find_observation_message_index,
    last_token_of_final_assistant,
    last_token_of_message_content,
)
from stage2.trajectories.schema import TrajectoryRecord, load_trajectories_from_dir


def rel_pos(step_index: int, n_steps: int) -> float:
    if n_steps <= 1:
        return 0.0
    return step_index / (n_steps - 1)


_THINK_RE = re.compile(r"<think>", re.IGNORECASE)


def trajectory_uses_thinking(records: list[TrajectoryRecord]) -> bool:
    """Detect whether the trajectories were generated with thinking mode ON.

    Under Qwen3 thinking-on served with a reasoning parser, the ``<think>`` text
    is split into the native ``assistant_message.reasoning_content`` field rather
    than left inline in ``content``; older/no-parser captures leave it inline. We
    treat a trajectory as thinking-on if *any* assistant turn carries a
    ``reasoning_content`` or an inline ``<think>`` block, checking both the
    per-step ``assistant_message`` / ``assistant_response`` and the assistant
    turns in ``messages_before_gen``.
    """
    for rec in records:
        for step in rec.steps:
            if step.assistant_message and step.assistant_message.get("reasoning_content"):
                return True
            if _THINK_RE.search(step.assistant_response or ""):
                return True
            for msg in step.messages_before_gen:
                if msg.get("role") == "assistant" and (
                    msg.get("reasoning_content")
                    or _THINK_RE.search(msg.get("content", "") or "")
                ):
                    return True
    return False


def assert_thinking_mode_matches(
    records: list[TrajectoryRecord], enable_thinking: bool
) -> None:
    """Fail loudly on a generation/projection thinking-mode mismatch.

    The projection replays each recorded turn through the chat template and reads
    the activation at its last token. If the template's thinking mode differs from
    the mode the trajectory was *generated* under, the tokenization (and thus the
    activation we read) no longer corresponds to what the model actually computed.
    That silently corrupts every projection, so we refuse to proceed either way.
    (Stage 2 runs thinking-ON by default as of 2026-07-17; legacy thinking-off
    data can still be projected with ``--no-enable-thinking``.)
    """
    generated_with_thinking = trajectory_uses_thinking(records)
    if generated_with_thinking and not enable_thinking:
        raise SystemExit(
            "Thinking-mode mismatch: trajectories were generated with thinking ON "
            "(reasoning_content / <think> blocks present) but projection is running "
            "with thinking OFF. Re-run with --enable-thinking so the chat template "
            "matches how the model actually generated."
        )
    if enable_thinking and not generated_with_thinking:
        raise SystemExit(
            "Thinking-mode mismatch: projection is running with thinking ON but the "
            "trajectories carry no reasoning_content / <think> blocks (generated "
            "thinking OFF). Re-run with --no-enable-thinking to match, or regenerate "
            "the trajectories with thinking ON."
        )


def _final_assistant_segment(full_text: str) -> str:
    """The rendered text of the last assistant turn (marker-delimited)."""
    marker = "<|im_start|>assistant"
    marker_pos = full_text.rfind(marker)
    if marker_pos < 0:
        return ""
    region_start = marker_pos + len(marker)
    end_pos = full_text.find("<|im_end|>", region_start)
    return full_text[region_start : end_pos if end_pos >= 0 else len(full_text)]


def assert_render_fidelity(full_text: str, assistant_message: dict) -> None:
    """Confirm the chat template re-rendered the native turn faithfully.

    The projection's correctness hinges on the Qwen3 template reproducing the
    generated token stream from the split ``reasoning_content`` / ``tool_calls``
    fields. If a template version drops the ``<think>`` block or mangles the tool
    call, the activation we read no longer matches what the model computed, so we
    fail loudly (the thinking-ON analogue of the old thinking-off guard). Run once
    per projection on the first structured turn.
    """
    segment = _final_assistant_segment(full_text)

    reasoning = (assistant_message.get("reasoning_content") or "").strip()
    if reasoning:
        probe = reasoning[: min(40, len(reasoning))]
        if probe and probe not in segment:
            raise SystemExit(
                "Render fidelity check failed: the assistant's reasoning_content is "
                "absent from the re-rendered turn. The Qwen3 chat template is not "
                "emitting the <think> block for completed turns, so the replayed "
                "tokens do not match generation. Verify the transformers/template "
                "version renders reasoning_content before trusting projections."
            )

    for tc in assistant_message.get("tool_calls") or []:
        name = (tc.get("function") or {}).get("name", "")
        if name and name not in segment:
            raise SystemExit(
                "Render fidelity check failed: tool call "
                f"{name!r} is absent from the re-rendered turn. The chat template is "
                "not reproducing tool_calls, so the replayed tokens do not match "
                "generation. Verify the template renders tool_calls (hermes-style) "
                "before trusting projections."
            )


def _encode_template(tokenizer, messages, enable_thinking: bool):
    text = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    return text, enc["input_ids"], offset_mapping


def _project_at_token(
    model,
    input_ids,
    layer: int,
    token_index: int,
    direction: np.ndarray,
    n_layers: int,
) -> float | None:
    capture = LayerActivationCapture(model, n_layers=n_layers)
    with torch.no_grad():
        model(input_ids=input_ids)
    layer_act = capture.get(layer)
    capture.remove()
    if layer_act is None:
        return None
    if layer_act.dim() == 3:
        layer_act = layer_act[0]
    if token_index >= layer_act.shape[0]:
        return None
    activation = layer_act[token_index].cpu().numpy()
    return project_activation(activation, direction)


def extract_rows_for_trajectory(
    record: TrajectoryRecord,
    model,
    tokenizer,
    *,
    layer: int,
    direction: np.ndarray,
    enable_thinking: bool,
    n_layers: int,
    device: torch.device,
    fidelity_state: dict | None = None,
) -> list[dict]:
    rows: list[dict] = []
    n_steps = record.n_steps

    for step in record.steps:
        i = step.step_index
        rp = rel_pos(i, n_steps)

        # The native assistant turn is authoritative for token-level replay; fall
        # back to a plain-content message for legacy thinking-off trajectories.
        assistant_msg = step.assistant_message or {
            "role": "assistant",
            "content": step.assistant_response,
        }
        if step.assistant_message is not None or step.assistant_response.strip():
            messages = list(step.messages_before_gen) + [assistant_msg]
            text, input_ids_list, offset_mapping = _encode_template(
                tokenizer, messages, enable_thinking
            )
            # Verify template fidelity once on the first structured turn: the read
            # position is only meaningful if the render reproduces generation.
            if (
                fidelity_state is not None
                and not fidelity_state.get("checked")
                and step.assistant_message is not None
            ):
                assert_render_fidelity(text, step.assistant_message)
                fidelity_state["checked"] = True
            span = last_token_of_final_assistant(text, offset_mapping)
            if span is not None:
                input_ids = torch.tensor([input_ids_list], device=device)
                proj = _project_at_token(
                    model, input_ids, layer, span.token_index, direction, n_layers
                )
                if proj is not None:
                    rows.append(
                        {
                            "trajectory_id": record.trajectory_id,
                            "outcome": record.outcome,
                            "step_index": i,
                            "n_steps": n_steps,
                            "rel_pos": rp,
                            "projection": proj,
                            "token_type": "reasoning",
                            "layer": layer,
                        }
                    )

        if i + 1 < n_steps and step.observation and step.observation.strip():
            next_step = record.steps[i + 1]
            obs_msg_idx = find_observation_message_index(
                next_step.messages_before_gen,
                step.observation,
            )
            if obs_msg_idx is not None:
                obs_content = next_step.messages_before_gen[obs_msg_idx]["content"]
                text, input_ids_list, offset_mapping = _encode_template(
                    tokenizer,
                    next_step.messages_before_gen,
                    enable_thinking,
                )
                span = last_token_of_message_content(
                    text, obs_content, offset_mapping
                )
                if span is not None:
                    input_ids = torch.tensor([input_ids_list], device=device)
                    proj = _project_at_token(
                        model, input_ids, layer, span.token_index, direction, n_layers
                    )
                    if proj is not None:
                        rows.append(
                            {
                                "trajectory_id": record.trajectory_id,
                                "outcome": record.outcome,
                                "step_index": i,
                                "n_steps": n_steps,
                                "rel_pos": rp,
                                "projection": proj,
                                "token_type": "tool_output",
                                "layer": layer,
                            }
                        )

    return rows


def run(
    traj_dir: Path,
    *,
    axis_path: Path,
    layer: int,
    model_name: str,
    enable_thinking: bool,
    dtype: str,
    n_layers: int,
    output_path: Path,
    check_thinking: bool = True,
) -> pd.DataFrame:
    records = load_trajectories_from_dir(traj_dir)
    if not records:
        raise FileNotFoundError(f"No normalized trajectories in {traj_dir}")

    if check_thinking:
        assert_thinking_mode_matches(records, enable_thinking)

    direction = load_axis_direction(axis_path, layer=layer)

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device

    all_rows: list[dict] = []
    fidelity_state: dict = {"checked": False}
    for record in records:
        print(
            f"  {record.trajectory_id}: {record.n_steps} steps, outcome={record.outcome}",
            flush=True,
        )
        rows = extract_rows_for_trajectory(
            record,
            model,
            tokenizer,
            layer=layer,
            direction=direction,
            enable_thinking=enable_thinking,
            n_layers=n_layers,
            device=device,
            fidelity_state=fidelity_state,
        )
        all_rows.extend(rows)
        print(f"    -> {len(rows)} projection rows", flush=True)

    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}", flush=True)
    return df


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--traj-dir",
        type=Path,
        default=NORMALIZED_DIR,
        help="Directory of normalized trajectory JSON files",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=data_file("projections.parquet"),
    )
    ap.add_argument("--layer", type=int, default=defaults["layer"])
    ap.add_argument("--model", default=defaults["model"])
    ap.add_argument("--axis-path", type=Path, default=defaults["axis_path"])
    ap.add_argument("--dtype", default=defaults["dtype"])
    ap.add_argument("--n-layers", type=int, default=defaults["n_layers"])
    ap.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=defaults["enable_thinking"],
        help="Render the chat template with Qwen3 thinking mode. Defaults to the "
        "config value (thinking ON in Stage 2); use --no-enable-thinking to "
        "project legacy thinking-off trajectories.",
    )
    ap.add_argument(
        "--no-thinking-check",
        action="store_true",
        help="Bypass the generation/projection thinking-mode consistency check",
    )
    ap.add_argument(
        "--mock-axis",
        action="store_true",
        help="Use random unit axis (smoke test only)",
    )
    args = ap.parse_args()

    axis_path = args.axis_path
    if args.mock_axis:
        rng = np.random.default_rng(42)
        mock = rng.standard_normal((defaults["n_layers"], defaults["hidden_dim"]))
        mock = mock / np.linalg.norm(mock, axis=1, keepdims=True)
        axis_path = data_file("mock_value_axis.npy")
        np.save(axis_path, mock.astype(np.float32))
        print(f"Using mock axis at {axis_path}", flush=True)
    else:
        require_axis_path(axis_path)

    print(f"Extracting projections from {args.traj_dir}...", flush=True)
    run(
        args.traj_dir,
        axis_path=axis_path,
        layer=args.layer,
        model_name=args.model,
        enable_thinking=args.enable_thinking,
        dtype=args.dtype,
        n_layers=args.n_layers,
        output_path=args.output,
        check_thinking=not args.no_thinking_check,
    )


if __name__ == "__main__":
    main()
