#!/usr/bin/env bash
# Offline Stage 1 wiring test (no GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/stage1"

pip install -e . -q

python tests/fixtures/icrl_mock.py
python tests/fixtures/fake_activations.py

python -m stage1.pipeline.run_gate --icrl data/mock_icrl.json --skip-extract --skip-mock --mock-only

echo "Stage 1 wiring test passed."
