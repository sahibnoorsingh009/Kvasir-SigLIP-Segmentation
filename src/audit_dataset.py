from __future__ import annotations
import argparse, json, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from .config import load_config
from .utils import find_by_stem, read_ids

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-overlays", type=int, default=80)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    images = find_by_stem(data["image_dir"])
    masks = find_by_stem(data["mask_dir"])
    splits = {
        "internal_train": read_ids(data["internal_train_split"]),
        "internal_val": read_ids(data["internal_val_split"]),
        "official_test": read_ids(data["official_test_split"]),
    }
    all_ids = sorted(set().union(*map(set, splits.values())))
    out = Path(args.out_dir)
    overlay_dir = out / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    split_lookup = {x: name for name, ids in splits.items() for x in ids}
    rows = []
    for image_id in all_ids:
        if image_id not in images or image_id not in masks:
            rows.append({"image_id": image_id, "valid": False, "reason": "missing_file"})
            continue
        image = cv2.imread(str(images[image_id]))
        mask = cv2.imread(str(masks[image_id]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            rows.append({"image_id": image_id, "valid": False, "reason": "read_failure"})
            continue
        binary = (mask >= 128).astype(np.uint8)
        n_components, _ = cv2.connectedComponents(binary)
        rows.append({
            "image_id": image_id,
            "split": split_lookup[image_id],
            "valid": True,
            "image_height": image.shape[0],
            "image_width": image.shape[1],
            "mask_height": mask.shape[0],
            "mask_width": mask.shape[1],
            "same_shape": bool(image.shape[:2] == mask.shape[:2]),
            "foreground_fraction": float(binary.mean()),
            "connected_components": int(max(n_components - 1, 0)),
            "empty_mask": bool(binary.sum() == 0),
            "full_mask": bool(binary.sum() == binary.size),
        })

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dataset_audit.csv", index=False)
    valid = df[df["valid"] == True]
    summary = {
        "image_files_discovered": len(images),
        "mask_files_discovered": len(masks),
        "audited_ids": len(all_ids),
        "valid_pairs": int(valid.shape[0]),
        "invalid_pairs": int((df["valid"] == False).sum()),
        "shape_mismatches": int((valid["same_shape"] == False).sum()),
        "empty_masks": int(valid["empty_mask"].sum()),
        "full_masks": int(valid["full_mask"].sum()),
        "foreground_fraction_mean": float(valid["foreground_fraction"].mean()),
        "foreground_fraction_median": float(valid["foreground_fraction"].median()),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_overlaps": {
            "train_val": len(set(splits["internal_train"]) & set(splits["internal_val"])),
            "train_test": len(set(splits["internal_train"]) & set(splits["official_test"])),
            "val_test": len(set(splits["internal_val"]) & set(splits["official_test"])),
        },
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rng = random.Random(args.seed)
    sample_ids = rng.sample(list(valid["image_id"]), min(args.num_overlays, len(valid)))
    for image_id in sample_ids:
        image = cv2.imread(str(images[image_id]))
        mask = cv2.imread(str(masks[image_id]), cv2.IMREAD_GRAYSCALE)
        binary = (mask >= 128).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(str(overlay_dir / f"{image_id}.jpg"), overlay)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
