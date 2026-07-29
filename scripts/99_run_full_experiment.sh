#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/logs results/statistics

echo "=================================================="
echo "FULL KVASIR-SEG EXPERIMENT"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "=================================================="

nvidia-smi
df -h /workspace

echo ""
echo "===== 1. DATASET AUDIT ====="
bash scripts/03_audit.sh

echo ""
echo "===== 2. RESUNET — SEEDS 42, 43, 44 ====="
for seed in 42 43 44; do
    echo "Starting ResUNet seed ${seed}: $(date)"
    python -m src.train \
        --config configs/resunet_paper.yaml \
        --seed "${seed}"
done

echo ""
echo "===== 3. SIGLIP2 FROZEN — SEEDS 42, 43, 44 ====="
for seed in 42 43 44; do
    echo "Starting frozen SigLIP2 seed ${seed}: $(date)"
    python -m src.train \
        --config configs/siglip2_frozen.yaml \
        --seed "${seed}"
done

echo ""
echo "===== 4. SIGLIP2 PARTIAL — SEEDS 42, 43, 44 ====="
for seed in 42 43 44; do
    echo "Starting partial SigLIP2 seed ${seed}: $(date)"
    python -m src.train \
        --config configs/siglip2_partial.yaml \
        --seed "${seed}"
done

echo ""
echo "===== 5. SIGLIP2 FULL — SEEDS 42, 43, 44 ====="
for seed in 42 43 44; do
    echo "Starting full SigLIP2 seed ${seed}: $(date)"
    python -m src.train \
        --config configs/siglip2_full.yaml \
        --seed "${seed}"
done

echo ""
echo "===== 6. FINAL OFFICIAL-SPLIT EVALUATION ====="

for seed in 42 43 44; do
    python -m src.evaluate \
        --config configs/resunet_paper.yaml \
        --checkpoint "checkpoints/resunet_paper/seed_${seed}/best.pt" \
        --seed "${seed}"

    for mode in frozen partial full; do
        python -m src.evaluate \
            --config "configs/siglip2_${mode}.yaml" \
            --checkpoint "checkpoints/siglip2_${mode}/seed_${seed}/best.pt" \
            --seed "${seed}"
    done
done

echo ""
echo "===== 7. PAIRED STATISTICAL ANALYSIS ====="

for mode in frozen partial full; do
    for seed in 42 43 44; do
        python -m src.statistical_analysis \
            --baseline "results/predictions/resunet_paper_seed${seed}.csv" \
            --candidate "results/predictions/siglip2_${mode}_seed${seed}.csv" \
            --name-a "ResUNet" \
            --name-b "SigLIP2-${mode}" \
            --out-dir "results/statistics/resunet_vs_siglip2_${mode}_seed${seed}" \
            --bootstrap 10000 \
            --permutations 10000 \
            --seed 2026
    done
done

echo ""
echo "=================================================="
echo "FULL EXPERIMENT FINISHED"
echo "Finished: $(date)"
echo "=================================================="
