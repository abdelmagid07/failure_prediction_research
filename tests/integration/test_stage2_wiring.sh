#!/usr/bin/env bash
# Offline Stage 2 wiring test (no GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/stage2"

pip install -e ../stage1 -q
pip install -e . -q

# --- Parsers: mini-swe-agent (primary) and SWE-agent (legacy) both wired. ---
python - <<'PY'
from pathlib import Path
from stage2.trajectories.parse_mini_swe_traj import parse_mini_swe_traj
from stage2.trajectories.parse_swe_traj import parse_swe_traj

mini = Path("tests/fixtures/sample_mini.traj.json")
rec = parse_mini_swe_traj(mini, outcome=1)
assert rec.trajectory_id == "example__repo-123", rec.trajectory_id
assert rec.n_steps == 3, rec.n_steps
# Reconstruction sanity: step 0's context is [system, user], observation flows in.
assert rec.steps[0].messages_before_gen[0]["role"] == "system"
assert rec.steps[0].observation.strip(), "step 0 should have an observation"
print(f"Parsed {mini.name} (mini-swe-agent): {rec.n_steps} steps")

legacy = Path("tests/fixtures/sample.traj")
rec2 = parse_swe_traj(legacy, outcome=1)
assert rec2.n_steps == 3, rec2.n_steps
print(f"Parsed {legacy.name} (swe-agent legacy): {rec2.n_steps} steps")
PY

# --- Ingest the primary path end to end, including the crash guard. ---
python - <<'PY'
from pathlib import Path
from tests.fixtures.trajectories_mock import write_mini_run_fixture
from stage2.trajectories.ingest_batch import ingest_batch

# Use gitignored roots (data/trajectories/, data/normalized/) so the smoke run
# leaves no untracked files behind.
run_dir = Path("data/trajectories/mini_run_smoke")
norm_dir = Path("data/normalized/mini_smoke")
write_mini_run_fixture(run_dir)
manifest = ingest_batch(run_dir, fmt="mini-swe-agent", output_dir=norm_dir)

assert manifest["format"] == "mini-swe-agent", manifest["format"]
assert manifest["n_ingested"] == 2, manifest["n_ingested"]
assert manifest["n_success"] == 1, manifest["n_success"]
assert manifest["n_failure"] == 1, manifest["n_failure"]
assert manifest["n_excluded_errors"] == 1, manifest  # the APIConnectionError stub
assert "APIConnectionError" in manifest["exit_status_histogram"], manifest["exit_status_histogram"]
assert (norm_dir / "mini_ok_1.json").exists()
assert not (norm_dir / "mini_crash_3.json").exists(), "crashed stub must not be ingested"
print("Ingest (mini-swe-agent) excluded the crash stub and labeled 1/1.")
PY

# --- Normalized -> mock projections -> analyses (schema-level coverage). ---
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
