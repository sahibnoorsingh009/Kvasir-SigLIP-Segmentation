from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from .config import load_config
from .dataset import KvasirSegDataset, Siglip2Collator, resunet_collate
from .engine import train_one_epoch, evaluate_loader
from .losses import build_loss
from .metrics import summarize_frame
from .models import build_model
from .transforms import build_transforms
from .utils import ensure_dir, seed_everything

def build_loaders(cfg, model):
    data = cfg["data"]
    image_size = int(data["image_size"])
    train_ds = KvasirSegDataset(
        data["image_dir"], data["mask_dir"], data["internal_train_split"],
        build_transforms(image_size, cfg.get("augmentation", {}), True),
    )
    val_ds = KvasirSegDataset(
        data["image_dir"], data["mask_dir"], data["internal_val_split"],
        build_transforms(image_size, cfg.get("augmentation", {}), False),
    )
    if cfg["model"]["name"] == "siglip2_unet":
        collate = Siglip2Collator(model.processor, int(cfg["model"].get("max_num_patches", 400)))
    else:
        collate = resunet_collate
    common = dict(
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(data.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader

def build_optimizer(model, cfg):
    t = cfg["training"]
    if cfg["model"]["name"] == "resunet":
        return torch.optim.NAdam(
            model.parameters(),
            lr=float(t["lr"]),
            betas=(0.9, 0.999),
            weight_decay=float(t.get("weight_decay", 0.0)),
        )
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder_params}
    decoder_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    groups = [{"params": decoder_params, "lr": float(t["decoder_lr"])}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": float(t["encoder_lr"])})
    return torch.optim.AdamW(groups, weight_decay=float(t.get("weight_decay", 0.01)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["training"].get("seed", 42))
    seed_everything(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this project.")
    device = torch.device("cuda")

    model = build_model(cfg).to(device)
    train_loader, val_loader = build_loaders(cfg, model)
    criterion = build_loss(cfg["training"]["loss"])
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["training"]["epochs"])
    )
    amp = bool(cfg["training"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and not torch.cuda.is_bf16_supported())

    exp = cfg["experiment_name"]
    ckpt_dir = ensure_dir(Path(cfg["output"]["checkpoint_root"]) / exp / f"seed_{seed}")
    hist_dir = ensure_dir(cfg["output"]["history_root"])
    best_score = -1.0
    best_epoch = -1
    patience = int(cfg["training"].get("patience", 30))
    min_delta = float(cfg["training"].get("min_delta", 1e-4))
    stale = 0
    history = []

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler if scaler.is_enabled() else None,
            amp, float(cfg["training"].get("grad_clip", 1.0))
        )
        val_loss, val_df = evaluate_loader(
            model, val_loader, criterion, device, float(cfg["training"]["threshold"]), amp
        )
        summary = summarize_frame(val_df)
        score = summary["mean_dice"]
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **summary}
        history.append(row)
        pd.DataFrame(history).to_csv(hist_dir / f"{exp}_seed{seed}.csv", index=False)

        print(json.dumps(row, indent=2))
        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "seed": seed,
                "epoch": epoch,
                "best_mean_dice": best_score,
            }, ckpt_dir / "best.pt")
        else:
            stale += 1
        scheduler.step()
        if stale >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    print(f"Best mean validation Dice: {best_score:.6f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
