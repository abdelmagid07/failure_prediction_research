"""Shim so value-axis experiment scripts find Stage 1 artifacts."""

from pathlib import Path

DEFAULT_LAYER = 21

# Research/stage1/data — value-axis lives at Research/value-axis/
_STAGE1_DATA = Path(__file__).resolve().parents[2] / "stage1" / "data"


def data_file(name: str) -> Path:
    return _STAGE1_DATA / name
