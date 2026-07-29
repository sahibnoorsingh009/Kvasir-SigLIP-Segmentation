#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data/kvasir-seg}"
mkdir -p "$DATA_ROOT"

python - <<'PY'
from pathlib import Path
import kagglehub
import shutil

destination = Path("/workspace/data/kvasir-seg")
downloaded = Path(kagglehub.dataset_download("debeshjha1/kvasirseg"))
print("KaggleHub cache:", downloaded)

target = destination / "download"
if target.exists():
    shutil.rmtree(target)
shutil.copytree(downloaded, target)
print("Copied dataset to:", target)

dirs = [p for p in target.rglob("*") if p.is_dir()]
for p in dirs:
    name = p.name.lower()
    if name in {"images", "image", "masks", "mask", "ground-truth", "ground_truth"}:
        print("Candidate folder:", p)
PY

if [ ! -d external/Kvasir-SEG-official ]; then
  mkdir -p external
  git clone --depth 1 https://github.com/DebeshJha/Kvasir-SEG.git external/Kvasir-SEG-official
fi

echo "Dataset and official repository downloaded."
echo "Run: python -m src.discover_data --root $DATA_ROOT/download"
