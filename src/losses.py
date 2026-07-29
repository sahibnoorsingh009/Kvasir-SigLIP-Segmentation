from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def soft_dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()

class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 1.0 - soft_dice_score(logits, targets)

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = 1.0 - soft_dice_score(logits, targets)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice

def build_loss(name: str) -> nn.Module:
    if name == "dice":
        return DiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"Unknown loss: {name}")
