#!/usr/bin/env python
"""Replay SWE-agent trajectories through Qwen3-8B and extract value-axis projections."""

from __future__ import annotations

import argparse
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
    last_token_of_message_content,
    last_token_of_suffix,
)
from stage2.trajectories.schema import TrajectoryRecord, load_trajectories_from_dir


def rel_pos(step_index: int, n_steps: int) -> float:
    if n_steps <= 1:
        return 0.0
    return step_index / (n_steps - 1)


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
) -> list[dict]:
    rows: list[dict] = []
    n_steps = record.n_steps

    for step in record.steps:
        i = step.step_index
        rp = rel_pos(i, n_steps)

        if step.assistant_response.strip():
            messages = list(step.messages_before_gen) + [
                {"role": "assistant", "content": step.assistant_response}
            ]
            text, input_ids_list, offset_mapping = _encode_template(
                tokenizer, messages, enable_thinking
            )
            span = last_token_of_suffix(text, step.assistant_response, offset_mapping)
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
) -> pd.DataFrame:
    records = load_trajectories_from_dir(traj_dir)
    if not records:
        raise FileNotFoundError(f"No normalized trajectories in {traj_dir}")

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
    ap.add_argument("--enable-thinking", action="store_true")
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
        enable_thinking=args.enable_thinking or defaults["enable_thinking"],
        dtype=args.dtype,
        n_layers=args.n_layers,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
