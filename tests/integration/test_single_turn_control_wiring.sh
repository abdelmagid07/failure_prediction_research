#!/usr/bin/env bash
# Offline single-turn-control wiring test (no GPU, no network).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/single_turn_control"

pip install -e ../stage1 -q
pip install -e ../stage2 -q
pip install -e . -q

# --- Unit tests: corruption determinism, render offsets, prepare wiring, stats. ---
python -m pytest tests -q

# --- End-to-end plumbing: mock problems -> manifest -> mock projections -> analyze. ---
python - <<'PY'
import json
from pathlib import Path

from single_turn_control.prepare import build_manifest_from_problems
from tests.fixtures.mock_problems import MOCK_PROBLEMS, FakeTokenizer

VARIANTS = ["buggy", "syntax_error", "shuffled", "obfuscated"]

manifest = build_manifest_from_problems(
    FakeTokenizer(), MOCK_PROBLEMS, variants=VARIANTS, enable_thinking=True,
)
assert len(manifest) == len(MOCK_PROBLEMS), manifest

out = Path("data/smoke_problems_manifest.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2))
print(f"Wrote {len(manifest)} problems -> {out}")
PY

python tests/fixtures/mock_projections.py \
  --output data/smoke_projections.parquet \
  --n-problems 20 --n-layers 8 --primary-layer 5

python -m single_turn_control.analyze \
  --projections data/smoke_projections.parquet \
  --output-dir data/smoke_report \
  --primary-layer 5 \
  --n-boot 200 --n-perm 200

test -f data/smoke_report/report.json
test -f data/smoke_report/layer_sweep.png

python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("data/smoke_report/report.json").read_text())
assert report["n_problems"] == 20, report["n_problems"]
pooled = report["primary"]["pooled"]
assert pooled["auroc"] > 0.8, pooled
assert pooled["ci_low"] > 0.5, pooled
print("Headline pooled AUROC:", pooled["auroc"], "CI:", (pooled["ci_low"], pooled["ci_high"]))
PY

echo "Single-turn-control wiring test passed."
