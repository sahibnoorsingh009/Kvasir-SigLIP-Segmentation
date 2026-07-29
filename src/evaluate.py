from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import KvasirSegDataset, Siglip2Collator, resunet_collate
from .engine import evaluate_loader, move_batch, model_forward
from .losses import build_loss
from .metrics import summarize_frame
from .models import build_model
from .transforms import build_transforms
from .utils import ensure_dir, seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qualitative-count", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)

    data = cfg["data"]
    ds = KvasirSegDataset(
        data["image_dir"], data["mask_dir"], data["official_test_split"],
        build_transforms(int(data["image_size"]), cfg.get("augmentation", {}), False),
    )
    collate = (
        Siglip2Collator(model.processor, int(cfg["model"].get("max_num_patches", 400)))
        if cfg["model"]["name"] == "siglip2_unet" else resunet_collate
    )
    loader = DataLoader(
        ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False,
        num_workers=int(data.get("num_workers", 4)), pin_memory=True, collate_fn=collate
    )
    criterion = build_loss(cfg["training"]["loss"])
    loss, df = evaluate_loader(
        model, loader, criterion, device, float(cfg["training"]["threshold"]),
        bool(cfg["training"].get("amp", True)) and device.type == "cuda",
    )
    exp = cfg["experiment_name"]
    pred_dir = ensure_dir(cfg["output"]["prediction_root"])
    pred_path = pred_dir / f"{exp}_seed{args.seed}.csv"
    df.to_csv(pred_path, index=False)
    summary = {"loss": loss, **summarize_frame(df)}
    (pred_dir / f"{exp}_seed{args.seed}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Predictions:", pred_path)

if __name__ == "__main__":
    main()
