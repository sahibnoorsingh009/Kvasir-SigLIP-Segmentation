from __future__ import annotations
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def locate(repo: Path, filename: str) -> Path:
    matches = list(repo.rglob(filename))
    matches = [p for p in matches if "Data-split" in str(p) or "data-split" in str(p).lower()]
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} below {repo}")
    return matches[0]

def read_ids(path: Path) -> list[str]:
    return [Path(x.strip()).stem for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--internal-val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    repo = Path(args.official_repo)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = locate(repo, "train.txt")
    val_path = locate(repo, "val.txt")
    official_train = read_ids(train_path)
    official_test = read_ids(val_path)

    overlap = set(official_train) & set(official_test)
    if overlap:
        raise ValueError(f"Official splits overlap: {sorted(overlap)[:5]}")

    internal_train, internal_val = train_test_split(
        official_train,
        test_size=args.internal_val_fraction,
        random_state=args.seed,
        shuffle=True,
    )
    internal_train = sorted(internal_train)
    internal_val = sorted(internal_val)

    write_ids(out / "official_train.txt", official_train)
    write_ids(out / "official_test.txt", official_test)
    write_ids(out / "internal_train.txt", internal_train)
    write_ids(out / "internal_val.txt", internal_val)

    print(f"Official train pool: {len(official_train)}")
    print(f"Internal train:      {len(internal_train)}")
    print(f"Internal val:        {len(internal_val)}")
    print(f"Official test:       {len(official_test)}")

if __name__ == "__main__":
    main()
