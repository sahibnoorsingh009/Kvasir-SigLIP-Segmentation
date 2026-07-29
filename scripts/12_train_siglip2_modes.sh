#!/usr/bin/env bash
set -euo pipefail
for cfg in configs/siglip2_frozen.yaml configs/siglip2_partial.yaml configs/siglip2_full.yaml; do
  for seed in 42 43 44; do
    python -m src.train --config "$cfg" --seed "$seed"
  done
done
