#!/usr/bin/env bash
set -euo pipefail

for seed in 42 43 44; do
  python -m src.evaluate \
    --config configs/resunet_paper.yaml \
    --checkpoint "checkpoints/resunet_paper/seed_${seed}/best.pt" \
    --seed "$seed"

  python -m src.evaluate \
    --config configs/siglip2_full.yaml \
    --checkpoint "checkpoints/siglip2_full/seed_${seed}/best.pt" \
    --seed "$seed"
done
