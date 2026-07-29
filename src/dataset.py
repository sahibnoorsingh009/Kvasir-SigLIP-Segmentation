from __future__ import annotations
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import find_by_stem, read_ids

class KvasirSegDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        split_file: str | Path,
        transform: Callable,
    ) -> None:
        self.images = find_by_stem(image_dir)
        self.masks = find_by_stem(mask_dir)
        self.ids = read_ids(split_file)
        self.transform = transform

        missing_images = [x for x in self.ids if x not in self.images]
        missing_masks = [x for x in self.ids if x not in self.masks]
        if missing_images or missing_masks:
            raise FileNotFoundError(
                f"Missing images={missing_images[:5]}, missing masks={missing_masks[:5]}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        image_id = self.ids[index]
        image = cv2.imread(str(self.images[image_id]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.masks[image_id]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Failed to read {image_id}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask >= 128).astype(np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        return {
            "id": image_id,
            "image_uint8": image,
            "mask": torch.from_numpy(mask.astype(np.float32))[None, ...],
        }

def resunet_collate(batch: list[dict]) -> dict:
    images = []
    masks = []
    ids = []
    for item in batch:
        image = torch.from_numpy(item["image_uint8"]).permute(2, 0, 1).float() / 255.0
        images.append(image)
        masks.append(item["mask"])
        ids.append(item["id"])
    return {"ids": ids, "pixel_values": torch.stack(images), "masks": torch.stack(masks)}

class Siglip2Collator:
    def __init__(self, processor, max_num_patches: int = 400) -> None:
        self.processor = processor
        self.max_num_patches = max_num_patches

    def __call__(self, batch: list[dict]) -> dict:
        images = [item["image_uint8"] for item in batch]
        processed = self.processor(
            images=images,
            return_tensors="pt",
            max_num_patches=self.max_num_patches,
        )
        return {
            "ids": [item["id"] for item in batch],
            "pixel_values": processed["pixel_values"],
            "pixel_attention_mask": processed.get("pixel_attention_mask"),
            "spatial_shapes": processed.get("spatial_shapes"),
            "masks": torch.stack([item["mask"] for item in batch]),
        }
