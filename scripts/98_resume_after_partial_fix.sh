#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/statistics results/predictions

echo "=================================================="
echo "RESUMED EXPERIMENT"
echo "Started: $(date)"
echo "=================================================="

echo "===== SIGLIP2 PARTIAL ====="
for seed in 42 43 44; do
    checkpoint="checkpoints/siglip2_partial/seed_${seed}/best.pt"

    if [[ -f "$checkpoint" ]]; then
        echo "Skipping partial seed ${seed}; checkpoint exists."
    else
        echo "Starting partial seed ${seed}: $(date)"
        python -m src.train \
          --config configs/siglip2_partial.yaml \
          --seed "${seed}"
    fi
done

echo "===== SIGLIP2 FULL ====="
for seed in 42 43 44; do
    checkpoint="checkpoints/siglip2_full/seed_${seed}/best.pt"

    if [[ -f "$checkpoint" ]]; then
        echo "Skipping full seed ${seed}; checkpoint exists."
    else
        echo "Starting full seed ${seed}: $(date)"
        python -m src.train \
          --config configs/siglip2_full.yaml \
          --seed "${seed}"
    fi
done

echo "===== FINAL EVALUATION ====="
for seed in 42 43 44; do
    for model in resunet_paper siglip2_frozen siglip2_partial siglip2_full; do
        checkpoint="checkpoints/${model}/seed_${seed}/best.pt"
        prediction="results/predictions/${model}_seed${seed}.csv"

        if [[ ! -f "$checkpoint" ]]; then
            echo "ERROR: Missing checkpoint: $checkpoint"
            exit 1
        fi

        if [[ -f "$prediction" ]]; then
            echo "Skipping evaluation ${model} seed ${seed}; result exists."
        else
            echo "Evaluating ${model} seed ${seed}"
            python -m src.evaluate \
              --config "configs/${model}.yaml" \
              --checkpoint "$checkpoint" \
              --seed "${seed}"
        fi
    done
done

echo "===== STATISTICAL ANALYSIS ====="
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

echo "=================================================="
echo "EXPERIMENT FINISHED"
echo "Finished: $(date)"
echo "=================================================="
