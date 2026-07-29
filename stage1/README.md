# Stage 1: Value axis reconstruction

Rebuild the value axis from ICRL conversations and verify held-out AUROC before any downstream work.

## Gate (Qwen3-8B default)

Default preset: L21 and L22 AUROC ≥ **0.87** (published 0.90 − 0.03) → freeze `data/value_axis.npy`.

```bash
pip install -e .

python -m stage1.icrl_gen.generate --n 300 --output data/icrl.json --resume
python -m stage1.pipeline.extract_activations --icrl data/icrl.json --force
python -m stage1.pipeline.run_gate --icrl data/icrl.json --skip-extract
```

Colab (8B): [notebooks/stage1_gpu_colab.ipynb](notebooks/stage1_gpu_colab.ipynb) (A100).

## Qwen3-32B faithful path (OpenRouter + Colab)

Same Jiang Appendix A recipe; extract with **`enable_thinking=true`** so the axis matches Azure Stage-2 trajs.

1. **Laptop — generate ICRL** (Claude Opus 4.6 via OpenRouter):

```bash
# .env: OPENROUTER_API_KEY=...  ICRL_BACKEND=openrouter
#       ICRL_MODEL=anthropic/claude-opus-4.6
pip install -e .   # needs openai

python -m stage1.icrl_gen.generate --n 2 --backend openrouter \
  --target-model Qwen3-32B --min-paragraphs 3 --max-paragraphs 8 \
  --output data/icrl_32b.json --resume   # smoke

python -m stage1.icrl_gen.generate --n 300 --backend openrouter \
  --target-model Qwen3-32B --min-paragraphs 3 --max-paragraphs 8 \
  --output data/icrl_32b.json --resume
```

2. **Colab A100 — three stages** (separate notebook cells; resume extract on Drive):
   [notebooks/stage1_gpu_colab_32b.ipynb](notebooks/stage1_gpu_colab_32b.ipynb)

```bash
# extract (GPU, long; resume-safe — skips existing .npz)
python -m stage1.pipeline.run_gate --preset qwen32b --stage extract \
  --icrl data/icrl_32b.json --activations-dir /path/to/activations_32b

# build axis (no model load)
python -m stage1.pipeline.run_gate --preset qwen32b --stage build \
  --activations-dir /path/to/activations_32b

# gate / AUROC (no model load; needs value_axis_32b.npy)
python -m stage1.pipeline.run_gate --preset qwen32b --stage gate \
  --activations-dir /path/to/activations_32b
```

`--stage all` still runs extract→build→gate in one process. Artifacts (do not overwrite 8B):
`value_axis_32b.npy` (64×5120), `axis_manifest_32b.json`, AUROC plot.
Primary layer = mid–late held-out AUROC argmax; gate ≥ **0.90**.

## Dev preset

For local Qwen ICRL without an API key:

```bash
python -m stage1.icrl_gen.generate --n 100 --backend local_qwen \
  --output data/icrl_proxy.json --resume --syntactic-only

python -m stage1.pipeline.extract_activations \
  --icrl data/icrl_proxy.json --activations-dir data/activations_proxy --force

python -m stage1.pipeline.run_gate --preset dev --icrl data/icrl_proxy.json --skip-extract
```

Writes `data/value_axis_proxy.npy` (threshold 0.75). Does not overwrite the default axis.

## Offline wiring test

```bash
bash ../tests/integration/test_stage1_wiring.sh
```

## Layout

```
stage1/
  config/           defaults.yaml, presets/{dev,qwen32b}.yaml, criteria.json, split.json
  stage1/
    icrl_gen/       ICRL generation (Anthropic, OpenRouter, or local Qwen)
    pipeline/       extract, build_axis, eval_auroc, run_gate
    common/         hooks, chat template, paths
  notebooks/        stage1_gpu_colab.ipynb, stage1_gpu_colab_32b.ipynb
  tests/fixtures/   offline mock data
```

## Debugging a failed gate

1. Boundary labels (`icrl/boundaries.py`)
2. Chat template / `enable_thinking`
3. Layer index and activation cache
4. Train/held-out split and ICRL quality
