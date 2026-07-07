#!/usr/bin/env python
"""Extract per-token layer activations for first post-discovery assistant turn."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage1.common.chat import apply_chat_template
from stage1.common.config import load_defaults
from stage1.common.hooks import LayerActivationCapture, get_transformer_num_layers
from stage1.common.paths import ACTIVATIONS_DIR, data_file
from stage1.icrl.boundaries import (
    build_messages_up_to_turn,
    compute_token_spans,
    get_first_post_discovery_turn,
)
from stage1.icrl.schema import load_conversations


def extract_one(
    model,
    tokenizer,
    conv,
    *,
    n_layers: int,
    enable_thinking: bool,
    device: torch.device,
) -> dict | None:
    post_turn = get_first_post_discovery_turn(conv)
    if post_turn is None or conv.satisfying_char_start is None:
        return None

    end_idx = conv.first_post_discovery_turn_idx
    messages = build_messages_up_to_turn(conv, end_idx)

    full_text = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = [(int(a), int(b)) for a, b in enc["offset_mapping"]]

    spans = compute_token_spans(
        full_text,
        offset_mapping,
        post_turn,
        conv.satisfying_char_start,
    )
    if spans is None:
        return None

    input_ids = torch.tensor([enc["input_ids"]], device=device)
    capture = LayerActivationCapture(model, n_layers=n_layers)
    with torch.no_grad():
        model(input_ids=input_ids)
    layer_acts = capture.all_layers(n_layers).numpy()  # (L, seq, hidden)

    token_labels = np.full(len(enc["input_ids"]), -1, dtype=np.int8)
    for i in spans.pre_indices:
        token_labels[i] = 0
    for i in spans.post_indices:
        token_labels[i] = 1

    capture.remove()
    return {
        "layer_activations": layer_acts.astype(np.float16),
        "token_labels": token_labels,
        "pre_indices": np.array(spans.pre_indices, dtype=np.int32),
        "post_indices": np.array(spans.post_indices, dtype=np.int32),
        "criterion_id": conv.criterion_id,
        "conv_id": conv.conv_id,
    }


def run(
    icrl_path: Path,
    *,
    model_name: str,
    n_layers: int,
    enable_thinking: bool,
    dtype: str,
    force: bool = False,
    activations_dir: Path | None = None,
) -> list[Path]:
    out_dir = activations_dir or ACTIVATIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    conversations = load_conversations(icrl_path)

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

    actual_layers = get_transformer_num_layers(model)
    if n_layers != actual_layers:
        print(
            f"Note: config n_layers={n_layers} but model has {actual_layers} layers; using {actual_layers}.",
            flush=True,
        )
        n_layers = actual_layers

    written = []
    for conv in conversations:
        out_path = out_dir / f"{conv.conv_id}.npz"
        if out_path.exists() and not force:
            written.append(out_path)
            continue

        result = extract_one(
            model,
            tokenizer,
            conv,
            n_layers=n_layers,
            enable_thinking=enable_thinking,
            device=device,
        )
        if result is None:
            print(f"  SKIP {conv.conv_id}: could not compute token spans", flush=True)
            continue

        np.savez_compressed(out_path, **result)
        written.append(out_path)
        print(f"  saved {conv.conv_id} -> {out_path}", flush=True)

    return written


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--icrl", type=Path, default=data_file("mock_icrl.json"))
    ap.add_argument("--model", default=defaults["model"])
    ap.add_argument("--n-layers", type=int, default=defaults["n_layers"])
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--dtype", default=defaults["dtype"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--activations-dir",
        type=Path,
        default=None,
        help="Output directory for .npz caches (default: data/activations)",
    )
    args = ap.parse_args()

    if not args.icrl.exists():
        raise SystemExit(f"ICRL file not found: {args.icrl}")

    act_dir = args.activations_dir or ACTIVATIONS_DIR
    print(f"Extracting activations from {len(load_conversations(args.icrl))} conversations...", flush=True)
    paths = run(
        args.icrl,
        model_name=args.model,
        n_layers=args.n_layers,
        enable_thinking=args.enable_thinking,
        dtype=args.dtype,
        force=args.force,
        activations_dir=act_dir,
    )
    print(f"Done. {len(paths)} activation files in {act_dir}", flush=True)


if __name__ == "__main__":
    main()
