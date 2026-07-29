from __future__ import annotations

import argparse
import json
from functools import reduce
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


METHODS = [
    "resunet_paper",
    "siglip2_frozen",
    "siglip2_partial",
    "siglip2_full",
]

METHOD_LABELS = {
    "resunet_paper": "ResUNet",
    "siglip2_frozen": "SigLIP2 Frozen",
    "siglip2_partial": "SigLIP2 Partial",
    "siglip2_full": "SigLIP2 Full",
}

SEEDS = [42, 43, 44]


def read_ids(path: str | Path) -> list[str]:
    return [
        Path(line.strip()).stem
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def discover_files(directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    files: dict[str, Path] = {}

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            files[path.stem] = path

    return files


def paired_bootstrap(
    differences: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(
        0,
        len(differences),
        size=(repetitions, len(differences)),
    )

    bootstrap_means = differences[indices].mean(axis=1)

    return (
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    )


def single_method_bootstrap(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(
        0,
        len(values),
        size=(repetitions, len(values)),
    )

    bootstrap_means = values[indices].mean(axis=1)

    return (
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    )


def paired_permutation_test(
    differences: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(float(differences.mean()))

    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(repetitions, len(differences)),
    )

    null_statistics = abs((signs * differences).mean(axis=1))

    return float(
        (1 + np.sum(null_statistics >= observed))
        / (repetitions + 1)
    )


def load_method_predictions(
    prediction_dir: Path,
    method: str,
) -> pd.DataFrame:
    frames = []

    for seed in SEEDS:
        path = prediction_dir / f"{method}_seed{seed}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)

        required = {
            "image_id",
            "dice",
            "iou",
            "precision",
            "recall",
            "specificity",
        }

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        frame = frame[
            [
                "image_id",
                "dice",
                "iou",
                "precision",
                "recall",
                "specificity",
            ]
        ].copy()

        frame = frame.rename(
            columns={
                metric: f"{metric}_{method}_{seed}"
                for metric in [
                    "dice",
                    "iou",
                    "precision",
                    "recall",
                    "specificity",
                ]
            }
        )

        frames.append(frame)

    merged = reduce(
        lambda left, right: left.merge(
            right,
            on="image_id",
            validate="one_to_one",
        ),
        frames,
    )

    for metric in [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
    ]:
        seed_columns = [
            f"{metric}_{method}_{seed}"
            for seed in SEEDS
        ]

        merged[f"{metric}_{method}_mean"] = merged[
            seed_columns
        ].mean(axis=1)

        merged[f"{metric}_{method}_seed_sd"] = merged[
            seed_columns
        ].std(axis=1, ddof=1)

    return merged


def calculate_foreground_fractions(
    ids: list[str],
    masks: dict[str, Path],
) -> pd.DataFrame:
    rows = []

    for image_id in ids:
        if image_id not in masks:
            raise FileNotFoundError(
                f"Mask not found for image ID: {image_id}"
            )

        mask = cv2.imread(
            str(masks[image_id]),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {masks[image_id]}"
            )

        binary = mask >= 128

        rows.append(
            {
                "image_id": image_id,
                "foreground_fraction": float(binary.mean()),
                "foreground_pixels": int(binary.sum()),
                "total_pixels": int(binary.size),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction-dir",
        default="results/predictions",
    )

    parser.add_argument(
        "--mask-dir",
        default=(
            "/workspace/data/kvasir-seg/download/"
            "Kvasir-SEG/Kvasir-SEG/masks"
        ),
    )

    parser.add_argument(
        "--official-train-split",
        default="splits/official_train.txt",
    )

    parser.add_argument(
        "--official-test-split",
        default="splits/official_test.txt",
    )

    parser.add_argument(
        "--out-dir",
        default="results/final_analysis",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    method_frames = {
        method: load_method_predictions(
            prediction_dir,
            method,
        )
        for method in METHODS
    }

    combined = method_frames[METHODS[0]]

    for method in METHODS[1:]:
        combined = combined.merge(
            method_frames[method],
            on="image_id",
            validate="one_to_one",
        )

    combined.to_csv(
        out_dir / "per_image_three_seed_aggregates.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Method-level confidence intervals
    # ----------------------------------------------------------

    method_summary_rows = []

    for method in METHODS:
        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_images": len(combined),
        }

        for metric in [
            "dice",
            "iou",
            "precision",
            "recall",
            "specificity",
        ]:
            values = combined[
                f"{metric}_{method}_mean"
            ].to_numpy()

            ci_low, ci_high = single_method_bootstrap(
                values,
                args.bootstrap,
                rng,
            )

            row[f"mean_{metric}"] = float(values.mean())
            row[f"median_{metric}"] = float(
                np.median(values)
            )
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high

        seed_dice_means = [
            float(
                combined[
                    f"dice_{method}_{seed}"
                ].mean()
            )
            for seed in SEEDS
        ]

        row["seed_mean_dice_sd"] = float(
            np.std(seed_dice_means, ddof=1)
        )

        row["mean_per_image_seed_sd"] = float(
            combined[
                f"dice_{method}_seed_sd"
            ].mean()
        )

        row["failure_rate_dice_lt_0_10"] = float(
            (
                combined[
                    f"dice_{method}_mean"
                ] < 0.10
            ).mean()
        )

        method_summary_rows.append(row)

    method_summary = pd.DataFrame(method_summary_rows)

    method_summary = method_summary.sort_values(
        "mean_dice",
        ascending=False,
    )

    method_summary.to_csv(
        out_dir / "method_summary_with_ci.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Primary aggregate paired comparison
    # ----------------------------------------------------------

    primary_rows = []

    baseline_method = "resunet_paper"

    for candidate_method in [
        "siglip2_frozen",
        "siglip2_partial",
        "siglip2_full",
    ]:
        for metric in [
            "dice",
            "iou",
            "precision",
            "recall",
            "specificity",
        ]:
            baseline_values = combined[
                f"{metric}_{baseline_method}_mean"
            ].to_numpy()

            candidate_values = combined[
                f"{metric}_{candidate_method}_mean"
            ].to_numpy()

            differences = (
                candidate_values - baseline_values
            )

            ci_low, ci_high = paired_bootstrap(
                differences,
                args.bootstrap,
                rng,
            )

            p_value = paired_permutation_test(
                differences,
                args.permutations,
                rng,
            )

            primary_rows.append(
                {
                    "baseline": baseline_method,
                    "candidate": candidate_method,
                    "metric": metric,
                    "n_images": len(differences),
                    "baseline_mean": float(
                        baseline_values.mean()
                    ),
                    "candidate_mean": float(
                        candidate_values.mean()
                    ),
                    "mean_difference": float(
                        differences.mean()
                    ),
                    "median_difference": float(
                        np.median(differences)
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "permutation_p": p_value,
                    "ci_excludes_zero": bool(
                        ci_low > 0 or ci_high < 0
                    ),
                    "practical_dice_improvement": (
                        bool(ci_low >= 0.02)
                        if metric == "dice"
                        else None
                    ),
                }
            )

    primary_results = pd.DataFrame(primary_rows)

    primary_results.to_csv(
        out_dir / "aggregate_paired_comparisons.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Lesion-size thresholds from official training masks
    # ----------------------------------------------------------

    masks = discover_files(args.mask_dir)

    train_ids = read_ids(args.official_train_split)
    test_ids = read_ids(args.official_test_split)

    train_size_df = calculate_foreground_fractions(
        train_ids,
        masks,
    )

    test_size_df = calculate_foreground_fractions(
        test_ids,
        masks,
    )

    lower_threshold = float(
        train_size_df[
            "foreground_fraction"
        ].quantile(1 / 3)
    )

    upper_threshold = float(
        train_size_df[
            "foreground_fraction"
        ].quantile(2 / 3)
    )

    def assign_size(value: float) -> str:
        if value <= lower_threshold:
            return "small"
        if value <= upper_threshold:
            return "medium"
        return "large"

    test_size_df["lesion_size_group"] = test_size_df[
        "foreground_fraction"
    ].apply(assign_size)

    combined = combined.merge(
        test_size_df,
        on="image_id",
        validate="one_to_one",
    )

    combined.to_csv(
        out_dir / "per_image_with_lesion_size.csv",
        index=False,
    )

    lesion_rows = []

    for size_group in [
        "small",
        "medium",
        "large",
    ]:
        subgroup = combined[
            combined["lesion_size_group"] == size_group
        ].copy()

        for candidate_method in [
            "siglip2_frozen",
            "siglip2_partial",
            "siglip2_full",
        ]:
            baseline_values = subgroup[
                "dice_resunet_paper_mean"
            ].to_numpy()

            candidate_values = subgroup[
                f"dice_{candidate_method}_mean"
            ].to_numpy()

            differences = (
                candidate_values - baseline_values
            )

            ci_low, ci_high = paired_bootstrap(
                differences,
                args.bootstrap,
                rng,
            )

            lesion_rows.append(
                {
                    "lesion_size_group": size_group,
                    "candidate": candidate_method,
                    "n_images": len(subgroup),
                    "foreground_fraction_min": float(
                        subgroup[
                            "foreground_fraction"
                        ].min()
                    ),
                    "foreground_fraction_median": float(
                        subgroup[
                            "foreground_fraction"
                        ].median()
                    ),
                    "foreground_fraction_max": float(
                        subgroup[
                            "foreground_fraction"
                        ].max()
                    ),
                    "resunet_mean_dice": float(
                        baseline_values.mean()
                    ),
                    "candidate_mean_dice": float(
                        candidate_values.mean()
                    ),
                    "mean_dice_difference": float(
                        differences.mean()
                    ),
                    "median_dice_difference": float(
                        np.median(differences)
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    lesion_results = pd.DataFrame(lesion_rows)

    lesion_results.to_csv(
        out_dir / "lesion_size_subgroup_analysis.csv",
        index=False,
    )

    thresholds = {
        "threshold_source": "official training masks",
        "small_upper": lower_threshold,
        "medium_upper": upper_threshold,
    }

    (
        out_dir / "lesion_size_thresholds.json"
    ).write_text(
        json.dumps(thresholds, indent=2),
        encoding="utf-8",
    )

    # ----------------------------------------------------------
    # Seed correlations
    # ----------------------------------------------------------

    correlation_rows = []

    for method in METHODS:
        dice_columns = [
            f"dice_{method}_{seed}"
            for seed in SEEDS
        ]

        correlation_matrix = combined[
            dice_columns
        ].corr()

        correlation_matrix.to_csv(
            out_dir
            / f"{method}_seed_dice_correlations.csv"
        )

        pairwise_values = []

        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                correlation = float(
                    correlation_matrix.iloc[i, j]
                )

                pairwise_values.append(correlation)

                correlation_rows.append(
                    {
                        "method": method,
                        "seed_a": SEEDS[i],
                        "seed_b": SEEDS[j],
                        "dice_correlation": correlation,
                    }
                )

        correlation_rows.append(
            {
                "method": method,
                "seed_a": "mean",
                "seed_b": "mean",
                "dice_correlation": float(
                    np.mean(pairwise_values)
                ),
            }
        )

    seed_correlations = pd.DataFrame(correlation_rows)

    seed_correlations.to_csv(
        out_dir / "seed_agreement.csv",
        index=False,
    )

    # Most unstable cases
    instability_rows = []

    for method in METHODS:
        method_instability = combined[
            [
                "image_id",
                f"dice_{method}_mean",
                f"dice_{method}_seed_sd",
                "foreground_fraction",
                "lesion_size_group",
            ]
        ].copy()

        method_instability["method"] = method

        method_instability = method_instability.rename(
            columns={
                f"dice_{method}_mean": "mean_dice",
                f"dice_{method}_seed_sd": "seed_sd",
            }
        )

        instability_rows.append(method_instability)

    instability_df = pd.concat(
        instability_rows,
        ignore_index=True,
    )

    instability_df.sort_values(
        "seed_sd",
        ascending=False,
    ).to_csv(
        out_dir / "per_image_seed_instability.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Win / tie / loss table
    # ----------------------------------------------------------

    win_loss_rows = []

    for candidate_method in [
        "siglip2_frozen",
        "siglip2_partial",
        "siglip2_full",
    ]:
        differences = (
            combined[
                f"dice_{candidate_method}_mean"
            ]
            - combined[
                "dice_resunet_paper_mean"
            ]
        )

        win_loss_rows.append(
            {
                "candidate": candidate_method,
                "clearly_better_gt_0_02": int(
                    (differences > 0.02).sum()
                ),
                "approximately_tied": int(
                    (
                        (differences >= -0.02)
                        & (differences <= 0.02)
                    ).sum()
                ),
                "clearly_worse_lt_minus_0_02": int(
                    (differences < -0.02).sum()
                ),
                "better_any_amount": int(
                    (differences > 0).sum()
                ),
                "worse_any_amount": int(
                    (differences < 0).sum()
                ),
                "mean_difference": float(
                    differences.mean()
                ),
                "median_difference": float(
                    differences.median()
                ),
            }
        )

    win_loss_df = pd.DataFrame(win_loss_rows)

    win_loss_df.to_csv(
        out_dir / "win_tie_loss_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Human-readable JSON summary
    # ----------------------------------------------------------

    full_dice_result = primary_results[
        (
            primary_results["candidate"]
            == "siglip2_full"
        )
        & (
            primary_results["metric"]
            == "dice"
        )
    ].iloc[0]

    full_row = method_summary[
        method_summary["method"]
        == "siglip2_full"
    ].iloc[0]

    resunet_row = method_summary[
        method_summary["method"]
        == "resunet_paper"
    ].iloc[0]

    final_summary = {
        "n_test_images": int(len(combined)),
        "resunet_mean_dice": float(
            resunet_row["mean_dice"]
        ),
        "resunet_dice_ci": [
            float(resunet_row["dice_ci_low"]),
            float(resunet_row["dice_ci_high"]),
        ],
        "siglip2_full_mean_dice": float(
            full_row["mean_dice"]
        ),
        "siglip2_full_dice_ci": [
            float(full_row["dice_ci_low"]),
            float(full_row["dice_ci_high"]),
        ],
        "full_vs_resunet_dice_difference": float(
            full_dice_result["mean_difference"]
        ),
        "full_vs_resunet_difference_ci": [
            float(full_dice_result["ci_low"]),
            float(full_dice_result["ci_high"]),
        ],
        "full_vs_resunet_permutation_p": float(
            full_dice_result["permutation_p"]
        ),
        "practical_threshold_met": bool(
            full_dice_result[
                "practical_dice_improvement"
            ]
        ),
        "lesion_size_thresholds": thresholds,
    }

    (
        out_dir / "final_summary.json"
    ).write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== METHOD SUMMARY ===")
    print(
        method_summary[
            [
                "method_label",
                "mean_dice",
                "dice_ci_low",
                "dice_ci_high",
                "mean_iou",
                "mean_precision",
                "mean_recall",
                "seed_mean_dice_sd",
                "failure_rate_dice_lt_0_10",
            ]
        ].to_string(index=False)
    )

    print("\n=== PRIMARY DICE COMPARISONS ===")
    print(
        primary_results[
            primary_results["metric"] == "dice"
        ][
            [
                "candidate",
                "baseline_mean",
                "candidate_mean",
                "mean_difference",
                "ci_low",
                "ci_high",
                "permutation_p",
                "practical_dice_improvement",
            ]
        ].to_string(index=False)
    )

    print("\n=== LESION-SIZE ANALYSIS: FULL SIGLIP2 ===")
    print(
        lesion_results[
            lesion_results["candidate"]
            == "siglip2_full"
        ][
            [
                "lesion_size_group",
                "n_images",
                "resunet_mean_dice",
                "candidate_mean_dice",
                "mean_dice_difference",
                "ci_low",
                "ci_high",
            ]
        ].to_string(index=False)
    )

    print("\n=== WIN / TIE / LOSS ===")
    print(win_loss_df.to_string(index=False))

    print(
        f"\nAll results saved to: {out_dir}"
    )


if __name__ == "__main__":
    main()
