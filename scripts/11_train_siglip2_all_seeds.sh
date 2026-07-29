#!/usr/bin/env bash
set -euo pipefail
for seed in 42 43 44; do
  python -m src.train --config configs/siglip2_full.yaml --seed "$seed"
done
