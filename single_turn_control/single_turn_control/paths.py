"""Path helpers for the single-turn coding control."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


def config_file(name: str) -> Path:
    return CONFIG_DIR / name


def data_file(name: str) -> Path:
    return DATA_DIR / name


def resolve_repo_path(relative: str) -> Path:
    """Resolve a config-declared path (e.g. axis_path) relative to this package's root."""
    return (ROOT / relative).resolve()
