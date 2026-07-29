from __future__ import annotations
import numpy as np
import pandas as pd
import torch

def per_image_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> dict[str, np.ndarray]:
    probs = torch.sigmoid(logits)
    pred = probs >= threshold
    true = targets >= 0.5
    dims = tuple(range(1, pred.ndim))

    tp = (pred & true).sum(dim=dims).float()
    fp = (pred & ~true).sum(dim=dims).float()
    fn = (~pred & true).sum(dim=dims).float()
    tn = (~pred & ~true).sum(dim=dims).float()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": dice.cpu().numpy(),
        "iou": iou.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "specificity": specificity.cpu().numpy(),
        "tp": tp.cpu().numpy(),
        "fp": fp.cpu().numpy(),
        "fn": fn.cpu().numpy(),
        "tn": tn.cpu().numpy(),
    }

def summarize_frame(df: pd.DataFrame) -> dict[str, float]:
    summary = {}
    for col in ["dice", "iou", "precision", "recall", "specificity"]:
        summary[f"mean_{col}"] = float(df[col].mean())
        summary[f"std_{col}"] = float(df[col].std(ddof=1))
        summary[f"median_{col}"] = float(df[col].median())

    tp, fp, fn, tn = [float(df[x].sum()) for x in ["tp", "fp", "fn", "tn"]]
    eps = 1e-7
    summary["global_dice"] = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    summary["global_iou"] = (tp + eps) / (tp + fp + fn + eps)
    summary["failure_rate_dice_lt_0_10"] = float((df["dice"] < 0.10).mean())
    return summary
