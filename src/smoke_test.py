from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader
from .config import load_config
from .dataset import KvasirSegDataset, Siglip2Collator, resunet_collate
from .models import build_model
from .transforms import build_transforms

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    model = build_model(cfg)
    data = cfg["data"]
    ds = KvasirSegDataset(
        data["image_dir"], data["mask_dir"], data["internal_val_split"],
        build_transforms(int(data["image_size"]), cfg.get("augmentation", {}), False)
    )
    collate = (
        Siglip2Collator(model.processor, int(cfg["model"].get("max_num_patches", 400)))
        if cfg["model"]["name"] == "siglip2_unet" else resunet_collate
    )
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate)))
    kwargs = {"pixel_values": batch["pixel_values"], "output_size": batch["masks"].shape[-2:]}
    if batch.get("pixel_attention_mask") is not None:
        kwargs["pixel_attention_mask"] = batch["pixel_attention_mask"]
    if batch.get("spatial_shapes") is not None:
        kwargs["spatial_shapes"] = batch["spatial_shapes"]
    with torch.no_grad():
        logits = model(**kwargs)
    print("pixel_values:", tuple(batch["pixel_values"].shape))
    print("spatial_shapes:", None if batch.get("spatial_shapes") is None else batch["spatial_shapes"])
    print("masks:", tuple(batch["masks"].shape))
    print("logits:", tuple(logits.shape))
    assert logits.shape == batch["masks"].shape
    print("Smoke test passed.")

if __name__ == "__main__":
    main()
