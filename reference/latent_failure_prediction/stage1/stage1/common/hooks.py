"""Activation hooks and cosine projection utilities."""

import numpy as np
import torch


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def cosine_projection(activations: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between each row of activations and a unit direction."""
    h = normalize_rows(activations.float())
    v = direction.float()
    v = v / v.norm().clamp(min=1e-8)
    return h @ v


class LayerActivationCapture:
    """Register forward hooks on all transformer layers and collect last forward pass."""

    def __init__(self, model, n_layers: int = 36):
        self._storage: dict[int, torch.Tensor] = {}
        self._handles = []

        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]

            def hook_fn(module, inp, output, idx=layer_idx):
                hs = output[0] if isinstance(output, tuple) else output
                self._storage[idx] = hs.detach()

            self._handles.append(layer.register_forward_hook(hook_fn))

    def clear(self):
        self._storage.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get(self, layer_idx: int) -> torch.Tensor | None:
        return self._storage.get(layer_idx)

    def all_layers(self, n_layers: int = 36) -> torch.Tensor:
        """Return (n_layers, seq_len, hidden_dim) for the cached forward pass."""
        layers = []
        for i in range(n_layers):
            if i not in self._storage:
                raise RuntimeError(f"Layer {i} activation not captured")
            t = self._storage[i]
            if t.dim() == 3:
                t = t[0]
            layers.append(t.cpu().float())
        return torch.stack(layers, dim=0)


def unit_direction(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def get_transformer_num_layers(model) -> int:
    """Return the number of decoder blocks in a HuggingFace causal LM."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    raise AttributeError("Could not infer num_hidden_layers from model")
