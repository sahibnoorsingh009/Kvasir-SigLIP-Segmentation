from __future__ import annotations

from pathlib import Path
from functools import reduce

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import KvasirSegDataset, Siglip2Collator, resunet_collate
from src.models import build_model
from src.transforms import build_transforms
from src.engine import move_batch, model_forward


SEEDS = [42, 43, 44]

REGRESSION_IDS = [
    "ck2bxw18mmz1k0725litqq2mc",
    "cju32zhbnc1oy0801iyv1ix6p",
    "cju7dubap2g0w0801fgl42mg9",
    "cju87xn2snfmv0987sc3d9xnq",
    "ck2bxlujamu330725szlc2jdu",
    "ck2bxiswtxuw80838qkisqjwz",
    "cju32qr9tbvsj08013pkpjenq",
    "cju87kbcen2av0987usezo8kn",
    "cju16b6ynq8e40988m8vx0xnj",
]


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = load_config(config_path)
    model = build_model(cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    return cfg, model


def predict_one(
    model,
    cfg,
    image_id: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = cfg["data"]

    tmp_split = Path("results/tmp_single_id.txt")
    tmp_split.write_text(image_id + "\n", encoding="utf-8")

    dataset = KvasirSegDataset(
        data["image_dir"],
        data["mask_dir"],
        tmp_split,
        build_transforms(
            int(data["image_size"]),
            cfg.get("augmentation", {}),
            training=False,
        ),
    )

    if cfg["model"]["name"] == "siglip2_unet":
        collate = Siglip2Collator(
            model.processor,
            int(cfg["model"].get("max_num_patches", 400)),
        )
    else:
        collate = resunet_collate

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    batch = next(iter(loader))
    original_rgb = batch["image_uint8"][0] if "image_uint8" in batch else None

    # Recover image directly from dataset path for display
    image_path = dataset.images[image_id]
    mask_path = dataset.masks[image_id]

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (int(data["image_size"]), int(data["image_size"])),
        interpolation=cv2.INTER_LINEAR,
    )

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(
        mask,
        (int(data["image_size"]), int(data["image_size"])),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = (mask >= 128).astype(np.uint8)

    batch = move_batch(batch, device)

    kwargs = {
        "pixel_values": batch["pixel_values"],
        "output_size": tuple(batch["masks"].shape[-2:]),
    }
    if batch.get("pixel_attention_mask") is not None:
        kwargs["pixel_attention_mask"] = batch["pixel_attention_mask"]
    if batch.get("spatial_shapes") is not None:
        kwargs["spatial_shapes"] = batch["spatial_shapes"]

    with torch.no_grad():
        logits = model(**kwargs)
        probs = torch.sigmoid(logits)
        pred = (probs >= float(cfg["training"]["threshold"])).float()

    pred = pred[0, 0].cpu().numpy().astype(np.uint8)

    return image, mask, pred


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def contour_overlay(image: np.ndarray, mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    overlay = image.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), thickness)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("results/regression_visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load one representative seed for direct visual comparison.
    # Seed 42 is used because the regression table above was seed-averaged,
    # while the visual panel needs one concrete prediction mask.
    res_cfg, res_model = load_model(
        "configs/resunet_paper.yaml",
        "checkpoints/resunet_paper/seed_42/best.pt",
        device,
    )

    sig_cfg, sig_model = load_model(
        "configs/siglip2_full.yaml",
        "checkpoints/siglip2_full/seed_42/best.pt",
        device,
    )

    summary_rows = []

    for image_id in REGRESSION_IDS:
        image, gt, res_pred = predict_one(
            res_model, res_cfg, image_id, device
        )
        _, _, sig_pred = predict_one(
            sig_model, sig_cfg, image_id, device
        )

        res_dice = dice_score(res_pred, gt)
        sig_dice = dice_score(sig_pred, gt)
        gain = sig_dice - res_dice

        fig = plt.figure(figsize=(18, 4.5))

        panels = [
            ("Original", image),
            ("Ground truth", contour_overlay(image, gt)),
            (f"ResUNet\nDice={res_dice:.3f}", contour_overlay(image, res_pred)),
            (f"SigLIP2 full\nDice={sig_dice:.3f}", contour_overlay(image, sig_pred)),
        ]

        for i, (title, panel) in enumerate(panels, start=1):
            ax = fig.add_subplot(1, 4, i)
            ax.imshow(panel)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(
            f"{image_id} | Dice difference SigLIP2 − ResUNet = {gain:+.3f}",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / f"{image_id}_comparison.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        summary_rows.append({
            "image_id": image_id,
            "resunet_dice_seed42": res_dice,
            "siglip2_full_dice_seed42": sig_dice,
            "dice_gain_seed42": gain,
        })

    pd.DataFrame(summary_rows).to_csv(
        out_dir / "visualized_regressions_summary.csv",
        index=False,
    )

    print(f"Saved comparison panels to: {out_dir}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
