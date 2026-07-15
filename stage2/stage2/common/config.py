"""Load Stage 2 YAML config."""

from typing import Any

import yaml

from stage2.common.paths import CONFIG_DIR, config_file, resolve_axis_path


def load_defaults() -> dict[str, Any]:
    with open(config_file("defaults.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["preset"] = "default"
    cfg["axis_path"] = resolve_axis_path(cfg["axis_path"])
    if "axis_manifest_path" in cfg:
        cfg["axis_manifest_path"] = resolve_axis_path(cfg["axis_manifest_path"])
    return cfg


def load_preset(name: str = "default") -> dict[str, Any]:
    if name == "default":
        return load_defaults()

    preset_path = CONFIG_DIR / "presets" / f"{name}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Unknown preset: {name} (no {preset_path})")

    with open(preset_path) as f:
        preset = yaml.safe_load(f)

    cfg = load_defaults()
    cfg.update(preset)
    cfg["axis_path"] = resolve_axis_path(cfg["axis_path"])
    if "axis_manifest_path" in cfg:
        cfg["axis_manifest_path"] = resolve_axis_path(cfg["axis_manifest_path"])
    return cfg
