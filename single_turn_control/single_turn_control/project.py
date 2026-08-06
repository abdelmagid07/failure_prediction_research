"""One forward pass over a prefilled (prompt + code) text -> per-layer projections.

Reuses ``stage1.common.hooks.LayerActivationCapture`` / ``cosine_projection``
(the same primitives ``stage1/pipeline/extract_activations.py`` and
``eval_auroc.py`` use) instead of the reference script's ad hoc
``per_token_cs`` hook. Layer indexing is this repo's own convention (hook
``model.model.layers[i]`` directly, axis row ``i`` — no off-by-one), *not*
the reference script's ``hook_idx = probe_layer - 1``.
"""

from __future__ import annotations

import torch

from stage1.common.hooks import LayerActivationCapture, cosine_projection


def code_token_span(
    offset_mapping: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int | None, int | None]:
    """Token index range covering [char_start, char_end) in the source text."""
    ts = te = None
    for i, (s, _e) in enumerate(offset_mapping):
        if s >= char_start and ts is None:
            ts = i
        if s < char_end:
            te = i + 1
    return ts, te


def run_forward(
    model,
    tokenizer,
    full_text: str,
    code_char_start: int,
    code_char_end: int,
    *,
    n_layers: int,
    directions: dict[int, torch.Tensor],
    device,
) -> dict[int, dict[str, float]] | None:
    """Per-layer {"mean": ..., "final": ..., "n_tokens": ...} over the code span.

    Returns ``None`` if the code span can't be located in the tokenization
    (mirrors the reference script's skip behavior for unmapped spans).
    """
    enc = tokenizer(
        full_text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False
    )
    offset_mapping = [(int(a), int(b)) for a, b in enc["offset_mapping"][0].tolist()]
    input_ids = enc["input_ids"].to(device)

    ts, te = code_token_span(offset_mapping, code_char_start, code_char_end)
    if ts is None or te is None or te <= ts:
        return None

    capture = LayerActivationCapture(model, n_layers=n_layers)
    with torch.no_grad():
        model(input_ids=input_ids)
    layer_acts = capture.all_layers(n_layers)  # (L, seq, H) cpu float32
    capture.remove()

    code_acts = layer_acts[:, ts:te, :]  # (L, n_code_tok, H)

    results: dict[int, dict[str, float]] = {}
    for layer, direction in directions.items():
        proj = cosine_projection(code_acts[layer], direction)  # (n_code_tok,)
        results[layer] = {
            "mean": float(proj.mean().item()),
            "final": float(proj[-1].item()),
            "n_tokens": int(proj.shape[0]),
        }
    return results
