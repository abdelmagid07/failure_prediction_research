"""Path helpers for Stage 2."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
TRAJ_DIR = DATA_DIR / "trajectories"
NORMALIZED_DIR = DATA_DIR / "normalized"


def config_file(name: str) -> Path:
    return CONFIG_DIR / name


def data_file(name: str) -> Path:
    return DATA_DIR / name


def resolve_axis_path(relative: str) -> Path:
    """Resolve axis path relative to stage2 root."""
    p = (ROOT / relative).resolve()
    return p


def require_axis_path(axis_path: Path) -> Path:
    if not axis_path.exists():
        raise FileNotFoundError(
            f"Value axis not found at {axis_path}. "
            "Complete Stage 1 gate first, or pass --mock-axis for smoke tests."
        )
    return axis_path
