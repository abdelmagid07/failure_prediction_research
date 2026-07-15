"""Load YAML/JSON config."""

import json
from pathlib import Path
from typing import Any

import yaml

from stage1.common.paths import CONFIG_DIR, DATA_DIR, config_file


def load_defaults() -> dict[str, Any]:
    with open(config_file("defaults.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["preset"] = "default"
    return cfg


def load_preset(name: str = "default") -> dict[str, Any]:
    """Load config for a named preset. ``default`` is the faithful Stage 1 path."""
    if name == "default":
        cfg = load_defaults()
        cfg["activations_dir"] = DATA_DIR / "activations"
        cfg["axis_path"] = DATA_DIR / "value_axis.npy"
        cfg["manifest_path"] = DATA_DIR / "axis_manifest.json"
        cfg["auroc_path"] = DATA_DIR / "auroc_by_layer.json"
        cfg["plot_path"] = DATA_DIR / "auroc_by_layer.png"
        cfg["icrl_path"] = DATA_DIR / "icrl.json"
        return cfg

    preset_path = CONFIG_DIR / "presets" / f"{name}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Unknown preset: {name} (no {preset_path})")

    with open(preset_path) as f:
        preset = yaml.safe_load(f)

    cfg = load_defaults()
    cfg.update(preset)

    cfg["activations_dir"] = DATA_DIR / preset.get("activations_subdir", "activations")
    cfg["axis_path"] = DATA_DIR / preset["axis_output"]
    cfg["manifest_path"] = DATA_DIR / preset["manifest_output"]
    cfg["auroc_path"] = DATA_DIR / preset["auroc_output"]
    cfg["plot_path"] = DATA_DIR / preset["plot_output"]
    cfg["icrl_path"] = DATA_DIR / preset.get("icrl_default", "icrl.json")
    return cfg


def load_criteria() -> list[dict]:
    with open(config_file("criteria.json")) as f:
        return json.load(f)


def load_split() -> dict[str, list[str]]:
    with open(config_file("split.json")) as f:
        return json.load(f)


def criteria_by_id() -> dict[str, dict]:
    return {c["id"]: c for c in load_criteria()}
