#!/usr/bin/env bash
set -euo pipefail
python -m src.audit_dataset \
  --config configs/resunet_paper.yaml \
  --out-dir results/audit \
  --num-overlays 80 \
  --seed 2026
