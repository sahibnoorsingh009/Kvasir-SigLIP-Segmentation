from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def paired_bootstrap(diff: np.ndarray, n: int, rng: np.random.Generator):
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    samples = diff[idx].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]), samples

def paired_permutation(diff: np.ndarray, n: int, rng: np.random.Generator):
    observed = abs(diff.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(diff)))
    null = abs((signs * diff).mean(axis=1))
    return float((1 + (null >= observed).sum()) / (n + 1))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--name-a", default="Baseline")
    p.add_argument("--name-b", default="Candidate")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    a = pd.read_csv(args.baseline)
    b = pd.read_csv(args.candidate)
    merged = a.merge(b, on="image_id", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("Files do not contain exactly the same image IDs.")

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "n_images": int(len(merged)),
        "model_a": args.name_a,
        "model_b": args.name_b,
    }

    for metric in ["dice", "iou", "precision", "recall", "specificity"]:
        diff = merged[f"{metric}_b"].to_numpy() - merged[f"{metric}_a"].to_numpy()
        ci, samples = paired_bootstrap(diff, args.bootstrap, rng)
        pvalue = paired_permutation(diff, args.permutations, rng)
        results[metric] = {
            "mean_a": float(merged[f"{metric}_a"].mean()),
            "mean_b": float(merged[f"{metric}_b"].mean()),
            "mean_difference_b_minus_a": float(diff.mean()),
            "median_difference_b_minus_a": float(np.median(diff)),
            "bootstrap_95_ci": [float(ci[0]), float(ci[1])],
            "paired_permutation_p": pvalue,
            "practical_improvement_ge_0_02": bool(ci[0] >= 0.02) if metric == "dice" else None,
        }

    merged["dice_difference_b_minus_a"] = merged["dice_b"] - merged["dice_a"]
    merged.sort_values("dice_difference_b_minus_a").to_csv(
        out / "paired_per_image_results.csv", index=False
    )
    (out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
