from __future__ import annotations
from pathlib import Path
import os, random
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_ids(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {p}")
    ids = []
    for line in p.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            ids.append(Path(value).stem)
    return ids

def find_by_stem(directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    out: dict[str, Path] = {}
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            if p.stem in out:
                raise ValueError(f"Duplicate stem '{p.stem}' in {directory}: {out[p.stem]} and {p}")
            out[p.stem] = p
    return out
