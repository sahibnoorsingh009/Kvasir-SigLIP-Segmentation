from __future__ import annotations
import albumentations as A
import cv2

def build_transforms(image_size: int, aug: dict, training: bool):
    if not training:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST),
        ])

    crop_min = float(aug.get("random_crop_scale_min", 0.85))
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(crop_min, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.45,
        ),
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=float(aug.get("horizontal_flip", 0.5))),
        A.VerticalFlip(p=float(aug.get("vertical_flip", 0.1))),
        A.Affine(
            scale=(1.0 - float(aug.get("scale_limit", 0.15)),
                   1.0 + float(aug.get("scale_limit", 0.15))),
            rotate=(-float(aug.get("rotate_limit", 20)),
                    float(aug.get("rotate_limit", 20))),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.55,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=float(aug.get("brightness_contrast", 0.25)),
            contrast_limit=float(aug.get("brightness_contrast", 0.25)),
            p=0.45,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.02, 0.10),
            hole_width_range=(0.02, 0.10),
            fill=0,
            fill_mask=None,
            p=float(aug.get("coarse_dropout", 0.25)),
        ),
    ])
