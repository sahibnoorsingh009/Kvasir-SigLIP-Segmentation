#!/usr/bin/env bash
set -euo pipefail
for seed in 42 43 44; do
  python -m src.train --config configs/resunet_paper.yaml --seed "$seed"
done
