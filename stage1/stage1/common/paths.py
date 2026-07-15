"""Path helpers and shared constants for Stage 1 reproduction."""

from pathlib import Path

DEFAULT_LAYER = 21

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
ACTIVATIONS_DIR = DATA_DIR / "activations"


def config_file(name: str) -> Path:
    return CONFIG_DIR / name


def data_file(name: str) -> Path:
    return DATA_DIR / name


def activation_file(conv_id: str, *, activations_dir: Path | None = None) -> Path:
    base = activations_dir or ACTIVATIONS_DIR
    return base / f"{conv_id}.npz"
