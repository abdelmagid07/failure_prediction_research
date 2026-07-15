# Stage 1: Value axis reconstruction

Rebuild the value axis on Qwen3-8B from ICRL conversations and verify held-out AUROC before any downstream work.

## Gate

Default preset: L21 and L22 AUROC ≥ **0.93** → freeze `data/value_axis.npy`.

```bash
pip install -e .

python -m stage1.icrl_gen.generate --n 300 --output data/icrl.json --resume
python -m stage1.pipeline.extract_activations --icrl data/icrl.json --force
python -m stage1.pipeline.run_gate --icrl data/icrl.json --skip-extract
```

Colab: [notebooks/stage1_gpu_colab.ipynb](notebooks/stage1_gpu_colab.ipynb) (A100).

## Dev preset

For local Qwen ICRL without the Anthropic API:

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
  config/           defaults.yaml, presets/dev.yaml, criteria.json, split.json
  stage1/
    icrl_gen/       ICRL generation (Anthropic or local Qwen)
    pipeline/       extract, build_axis, eval_auroc, run_gate
    common/         hooks, chat template, paths
  notebooks/        stage1_gpu_colab.ipynb
  tests/fixtures/   offline mock data
```

## Debugging a failed gate

1. Boundary labels (`icrl/boundaries.py`)
2. Chat template / `enable_thinking`
3. Layer index and activation cache
4. Train/held-out split and ICRL quality
