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
from stage2.extract.project_steps import trajectory_uses_thinking

mini = Path("tests/fixtures/sample_mini.traj.json")
rec = parse_mini_swe_traj(mini, outcome=1)
assert rec.trajectory_id == "example__repo-123", rec.trajectory_id
assert rec.n_steps == 3, rec.n_steps
# Identity fields (Chunk A): task_id separate from trajectory_id, seed defaults
# to None for a flat/single rollout, exit_status carried from info.
assert rec.task_id == "example__repo-123", rec.task_id
assert rec.seed is None, rec.seed
assert rec.exit_status == "Submitted", rec.exit_status
# A rollout seed suffixes the trajectory id but leaves the task id intact.
rec_seeded = parse_mini_swe_traj(mini, outcome=1, seed=3)
assert rec_seeded.trajectory_id == "example__repo-123__r3", rec_seeded.trajectory_id
assert rec_seeded.task_id == "example__repo-123", rec_seeded.task_id
assert rec_seeded.seed == 3, rec_seeded.seed
# Reconstruction sanity: step 0's context is [system, user], observation flows in.
assert rec.steps[0].messages_before_gen[0]["role"] == "system"
assert rec.steps[0].observation.strip(), "step 0 should have an observation"

# Thinking-ON shape: the native assistant turn carries reasoning + tool calls,
# and the flattened convenience string embeds the <think> block.
step0 = rec.steps[0]
assert step0.assistant_message is not None, "assistant_message must be preserved"
assert step0.assistant_message.get("reasoning_content"), "reasoning_content must be carried"
assert step0.assistant_message.get("tool_calls"), "tool_calls must be carried"
assert "<think>" in step0.assistant_response, "derived response should embed <think>"
# History stays native: a prior assistant turn keeps tool_calls, the observation
# is a tool-role message keyed by tool_call_id.
hist = rec.steps[1].messages_before_gen
assert any(m["role"] == "assistant" and m.get("tool_calls") for m in hist), hist
assert any(m["role"] == "tool" and m.get("tool_call_id") for m in hist), hist
# The projection guard must recognize these as thinking-on.
assert trajectory_uses_thinking([rec]), "guard should detect thinking-on"
print(f"Parsed {mini.name} (mini-swe-agent, thinking-on): {rec.n_steps} steps")

legacy = Path("tests/fixtures/sample.traj")
rec2 = parse_swe_traj(legacy, outcome=1)
assert rec2.n_steps == 3, rec2.n_steps
print(f"Parsed {legacy.name} (swe-agent legacy): {rec2.n_steps} steps")

# --- Chunk B readout geometry: G_t covers the whole assistant region and the
# mean-readout's last token equals the single-token robustness read. Uses a
# per-char offset mapping so it needs no tokenizer/GPU.
from stage2.extract.token_spans import (
    generated_token_indices,
    last_token_of_final_assistant,
)

full = (
    "<|im_start|>user\nhi<|im_end|>\n"
    "<|im_start|>assistant\n<think>reason</think>\nanswer\n"
    "<tool_call>{\"name\": \"bash\"}</tool_call><|im_end|>\n"
)
offsets = [(i, i + 1) for i in range(len(full))]  # one token per character
gt = generated_token_indices(full, offsets)
assert gt, "G_t must be non-empty on the thinking-on render"
region_start = full.rfind("<|im_start|>assistant") + len("<|im_start|>assistant")
region_end = full.find("<|im_end|>", region_start)
# Every generated-region character is covered, and nothing outside it.
assert min(gt) >= region_start, (min(gt), region_start)
assert max(gt) < region_end, (max(gt), region_end)
# The assistant region carries both the <think> block and the tool call.
covered = full[gt[0] : gt[-1] + 1]
assert "<think>" in covered and "<tool_call>" in covered, covered
# proj_final reads the last generated token == last element of G_t.
last_span = last_token_of_final_assistant(full, offsets)
assert last_span is not None and last_span.token_index == gt[-1], (last_span, gt[-1])
print(f"G_t covers {len(gt)} tokens; final-token read aligns with G_t[-1]")
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
# Identity fields (Chunk A) survive ingest into the manifest: task_id present,
# exit_status recorded, seed None for this flat (single-rollout) run dir.
by_id = {t["trajectory_id"]: t for t in manifest["trajectories"]}
assert by_id["mini_ok_1"]["task_id"] == "mini_ok_1", by_id["mini_ok_1"]
assert by_id["mini_ok_1"]["seed"] is None, by_id["mini_ok_1"]
assert by_id["mini_ok_1"]["exit_status"] == "Submitted", by_id["mini_ok_1"]
assert manifest["excluded_errors"][0]["exit_status"] == "APIConnectionError", manifest["excluded_errors"]
print("Ingest (mini-swe-agent) excluded the crash stub and labeled 1/1.")
PY

# --- Chunk C: multi-rollout ingest (r<seed>/ layout, per-rollout labels) + regen plan. ---
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from tests.fixtures.trajectories_mock import write_mini_rollout_fixture
from stage2.trajectories.ingest_batch import ingest_batch
import list_regens

run_dir = Path("data/trajectories/mini_rollout_smoke")
norm_dir = Path("data/normalized/mini_rollout_smoke")
write_mini_rollout_fixture(run_dir)
manifest = ingest_batch(run_dir, fmt="mini-swe-agent", output_dir=norm_dir)

by_id = {t["trajectory_id"]: t for t in manifest["trajectories"]}
# Seed recovered from the r<seed>/ directory layout.
assert by_id["roll_a__r0"]["seed"] == 0, by_id["roll_a__r0"]
assert by_id["roll_a__r1"]["seed"] == 1, by_id["roll_a__r1"]
assert by_id["roll_a__r0"]["task_id"] == "roll_a", by_id["roll_a__r0"]
# Per-rollout outcomes: roll_a resolves at seed 0, not at seed 1.
assert by_id["roll_a__r0"]["outcome"] == 1, by_id["roll_a__r0"]
assert by_id["roll_a__r1"]["outcome"] == 0, by_id["roll_a__r1"]
# Each rollout's own results.json was used.
assert len(manifest["results_paths"]) == 2, manifest["results_paths"]
# The seed-1 infra crash is excluded, and the regen planner assigns a fresh seed
# (0 and 1 are taken for roll_b -> 2).
excl = manifest["excluded_errors"]
assert excl and excl[0]["task_id"] == "roll_b" and excl[0]["seed"] == 1, excl
regens = list_regens.plan_regens(manifest)
assert regens and regens[0]["task_id"] == "roll_b" and regens[0]["fresh_seed"] == 2, regens
print("Multi-rollout ingest recovered seeds + per-rollout labels; regen plan fresh seed=2.")
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
test -f data/smoke_report/auroc_by_position.csv
test -f data/smoke_report/noise_by_token_type.png

# --- Chunk E: post-hoc confidence (dry-run) + internal-vs-elicited. ---
python -m stage2.elicit.confidence \
  --traj-dir data/normalized_smoke \
  --output data/confidence_smoke.parquet \
  --dry-run
python -m stage2.analyze.internal_vs_elicited \
  --projections data/projections_smoke.parquet \
  --confidence data/confidence_smoke.parquet \
  --output-dir data/smoke_report
test -f data/smoke_report/internal_vs_elicited.json

# --- Chunk F: fitted probes on a tiny synthetic activations.npz. ---
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
from stage2.probes.fit_probes import run as fit_probes

df = pd.read_parquet("data/projections_smoke.parquet")
# Fake activation vectors: signal = outcome along dim 0, noise elsewhere.
rng = np.random.default_rng(0)
n, h = len(df), 32
acts = rng.normal(0, 1, size=(n, h)).astype(np.float16)
acts[:, 0] = (df["outcome"].to_numpy() * 2 - 1).astype(np.float16)
path = Path("data/activations_smoke.npz")
np.savez_compressed(
    path,
    activations=acts,
    trajectory_id=df["trajectory_id"].to_numpy(),
    task_id=df["task_id"].to_numpy(),
    seed=np.array([-1 if s is None or (isinstance(s, float) and np.isnan(s)) else int(s)
                   for s in df["seed"]], dtype=np.int64),
    outcome=df["outcome"].to_numpy().astype(np.int64),
    step_index=df["step_index"].to_numpy().astype(np.int64),
    rel_pos=df["rel_pos"].to_numpy().astype(np.float64),
    layer=df["layer"].to_numpy().astype(np.int64),
)
fit_probes(path, output_dir=Path("data/smoke_probe"), n_bins=5, n_folds=2)
assert Path("data/smoke_probe/probe_auroc_grid.csv").exists()
print("Probe grid wrote OK")
PY

echo "Stage 2 wiring test passed."
