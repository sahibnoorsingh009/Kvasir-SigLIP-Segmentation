#!/usr/bin/env bash
set -euo pipefail
python -m src.prepare_splits \
  --official-repo external/Kvasir-SEG-official \
  --out-dir splits \
  --internal-val-fraction 0.10 \
  --seed 2026
