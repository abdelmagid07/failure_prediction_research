# Roadmap

Proposal: [PROPOSAL.tex](PROPOSAL.tex). Model: Qwen3-8B. Eval: SWE-bench.

**Compute:** Colab A100 serves Qwen via vLLM (tunnel URL). SWE-agent + Docker run locally and call that endpoint for every model step. GPU work also runs on Colab. 

Stage 1 is done for our purposes: value axis file created and got 0.87 AUROC on layers 21/22. We'll use that for pipeline work. We can try to get the original axis or ICRL data from the authors, to compare against their released axis if they publish one.

---

## Starting Status

**Have:** ICRL → extract → axis → gate (`stage1/`). Trajectory parse/ingest, per-step projection at L21 on the last reasoning token (`stage2/extract/project_steps.py`). Final-step AUROC, SNR-by-position, separation of results (`stage2/analyze/`). Colab notebooks for vLLM serve and projection. Mini-agent env for dry runs without Docker. Tests in `tests/integration/`.

**Don't have yet:** Real SWE-bench trajectories stored. Majority-class baseline in analysis output. Stage 3 curves and per-step elicited P(success). Multi-layer projection and layer-wise transfer AUROC. SWE-agent + Docker wired up for `run_pilot_batch.sh`.

The proposal metrics are separation curves, transfer AUROC by position, and internal vs elicited prediction.

---

## Week 1 — SWE-bench end-to-end

Get full trajectories working:

1. **Colab:** `serve_qwen_colab.ipynb` → copy tunnel URL (`MODEL_API_BASE`).
2. **Local (Docker):** `pip install sweagent`, set `MODEL_API_BASE`, run `run_pilot_batch.sh` with 3–5 instances from `config/pilot_instances.txt`.
3. Ingest with `results.json` for labels.
4. **Colab:** project via `project_and_analyze_colab.ipynb` or `project_steps.py` on A100.

Fix whatever breaks on real `.traj` files. 

Results: `normalized/` + `projections.parquet` from at least three SWE runs. 

---

## Week 2 — Final-step transfer

Run the approx. 20-instance pilot list. This is for proposal Stage 2: does the frozen axis separate resolved vs unresolved at the last step?

- Report N_success and N_failure; expand the instance list if successes are very thin.
- Add majority-class baseline to `final_step.py` and surface it in `analysis_report.json`.
- Once the pilot is clean, do the full 150+ instance generation. 

Output: `final_step_separation.png` and the first actual result of the paper. Label the result as clean, partial, or no transfer and continue.

---

## Week 3 — Projection vs position

Mean projection for success vs failure trajectories, binned by `rel_pos` (already in the parquet). AUROC at each bin.

- `separation_curve.py` — the main figure from the proposal.
- `transfer_auroc_by_position.py` — AUROC per bin vs majority class.
- Hook into `run_analyses.py` or a small `run_stage3.py`. 

---

## Week 4 — Elicited confidence

At each step, ask the model for P(eventual success) on the same trajectories. Compare that to the internal projection. 

- `stage2/elicit/confidence.py` — batch over saved trajectories via the Colab vLLM endpoint.
- `internal_vs_elicited.py` — AUROC by position for both signals.

---

## Week 5 — Layers

Project at more layers than L21 on Colab A100. See if transfer peaks where Stage 1 did (~21–22) or somewhere else.

- `--layers` on `project_steps.py`. Start with 17–25 if the full 36 is too slow.
- `layer_transfer.py`: per-layer AUROC and curves.
- Split half and half. Pick layers on half the trajectories, report on the rest.

---

## Week 6 — Figures and writing

Assemble the proposals main figures needed:

1. Stage 1 reconstruction AUROC (Done from noise testing)
2. Projection separation curve (week 3)
3. Transfer AUROC by position 
4. Internal vs elicited
5. Layer localization

Figures 1–3 are most essential; 4 and 5 are stretches.

---

## Commands

Stage 1 (already run):

```bash
cd stage1 && pip install -e .
python -m stage1.pipeline.run_gate --preset dev --icrl data/icrl_proxy.json --skip-extract
```

Stage 2 pipeline:

```bash
# Colab A100: serve_qwen_colab.ipynb → copy MODEL_API_BASE (keep tab open)

# Local machine (Docker + SWE-agent):
export MODEL_API_BASE="https://<tunnel>/v1"
cd stage2 && bash scripts/run_pilot_batch.sh config/pilot_instances.txt

# Ingest:
python -m stage2.trajectories.ingest_batch --traj-dir data/trajectories/run_<timestamp>

# Colab A100: projection
python -m stage2.extract.project_steps \
  --traj-dir data/normalized \
  --axis-path ../stage1/data/value_axis_proxy.npy

python -m stage2.analyze.run_analyses --projections data/projections.parquet
```

Not built yet: `run_stage3`, `elicit/confidence`, multi-layer `project_steps`.

Wiring tests: `bash tests/integration/test_stage1_wiring.sh` and `test_stage2_wiring.sh`.

Artifacts in `stage1/data/` and `stage2/data/` (gitignored). See [docs/setup.md](docs/setup.md).





