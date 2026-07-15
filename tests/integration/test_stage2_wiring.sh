#!/usr/bin/env bash
# Offline Stage 2 wiring test (no GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/stage2"

pip install -e ../stage1 -q
pip install -e . -q

python - <<'PY'
from pathlib import Path
from stage2.trajectories.parse_swe_traj import parse_swe_traj

sample = Path("tests/fixtures/sample.traj")
record = parse_swe_traj(sample, outcome=1)
assert record.n_steps == 3
print(f"Parsed {sample.name}: {record.n_steps} steps")
PY

python - <<'PY'
from pathlib import Path
from tests.fixtures.trajectories_mock import write_smoke_fixtures

paths = write_smoke_fixtures(
    Path("data/normalized_smoke"),
    sample_traj=Path("tests/fixtures/sample.traj"),
)
print(f"Wrote {len(paths)} normalized trajectories")
PY

python tests/fixtures/mock_projections.py \
  --traj-dir data/normalized_smoke \
  --output data/projections_smoke.parquet

python -m stage2.analyze.run_analyses \
  --projections data/projections_smoke.parquet \
  --output-dir data/smoke_report

test -f data/smoke_report/analysis_report.json
test -f data/smoke_report/final_step_separation.png
test -f data/smoke_report/noise_by_token_type.png
echo "Stage 2 wiring test passed."
