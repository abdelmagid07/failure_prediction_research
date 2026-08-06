"""Load single-turn-control YAML config."""

from typing import Any

import yaml

from single_turn_control.paths import config_file, resolve_repo_path


def load_defaults() -> dict[str, Any]:
    with open(config_file("defaults.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["axis_path"] = resolve_repo_path(cfg["axis_path"])
    cfg["axis_manifest_path"] = resolve_repo_path(cfg["axis_manifest_path"])
    return cfg
