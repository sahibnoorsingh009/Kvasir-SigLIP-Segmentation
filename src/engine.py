from __future__ import annotations
from contextlib import nullcontext
import pandas as pd
import torch
from tqdm import tqdm
from .metrics import per_image_metrics

def move_batch(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    for key in ["pixel_values", "pixel_attention_mask", "spatial_shapes", "masks"]:
        value = out.get(key)
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
    return out

def model_forward(model, batch: dict) -> torch.Tensor:
    kwargs = {
        "pixel_values": batch["pixel_values"],
        "output_size": tuple(batch["masks"].shape[-2:]),
    }
    if batch.get("pixel_attention_mask") is not None:
        kwargs["pixel_attention_mask"] = batch["pixel_attention_mask"]
    if batch.get("spatial_shapes") is not None:
        kwargs["spatial_shapes"] = batch["spatial_shapes"]
    return model(**kwargs)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp, grad_clip):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp else nullcontext()
        with context:
            logits = model_forward(model, batch)
            loss = criterion(logits, batch["masks"])
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        batch_size = batch["masks"].shape[0]
        total_loss += float(loss.detach()) * batch_size
        n += batch_size
    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, threshold, amp):
    model.eval()
    rows = []
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp else nullcontext()
        with context:
            logits = model_forward(model, batch)
            loss = criterion(logits, batch["masks"])
        metrics = per_image_metrics(logits.float(), batch["masks"].float(), threshold)
        for i, image_id in enumerate(batch["ids"]):
            row = {"image_id": image_id}
            for key, values in metrics.items():
                row[key] = float(values[i])
            rows.append(row)
        batch_size = batch["masks"].shape[0]
        total_loss += float(loss) * batch_size
        n += batch_size
    return total_loss / max(n, 1), pd.DataFrame(rows)
