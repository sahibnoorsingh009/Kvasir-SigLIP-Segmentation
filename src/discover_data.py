from __future__ import annotations
import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def count_images(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    candidates = []
    for p in root.rglob("*"):
        if p.is_dir():
            n = count_images(p)
            if n >= 500:
                candidates.append((n, p))
    for n, p in sorted(candidates, reverse=True)[:20]:
        print(f"{n:5d} files  {p}")

if __name__ == "__main__":
    main()
