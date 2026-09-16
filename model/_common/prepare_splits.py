#!/usr/bin/env python3
"""Shared utilities: resnet-aligned 5-fold splits + CLAM bags CSV (no leakage)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]  # /workspace
RESNET_SPLITS = WORKSPACE / "resnet" / "resnet50" / "splits"
LABEL_CSV = WORKSPACE / "data" / "sln-breast" / "target.csv"
PATCHES = WORKSPACE / "mil-set" / "patches_256"
SEED = 42


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(LABEL_CSV)
    df["slide_id"] = df["slide"].astype(str).str.replace(".svs", "", regex=False)
    df["label"] = df["target"].astype(int)
    return df[["slide_id", "label"]]


def sync_splits(out_dir: Path | None = None) -> Path:
    """Copy resnet splits verbatim for fair comparison."""
    out_dir = out_dir or (ROOT / "splits")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(RESNET_SPLITS.glob("fold_*.csv")):
        shutil.copy2(p, out_dir / p.name)
    shutil.copy2(RESNET_SPLITS / "summary.json", out_dir / "summary.json")
    meta = out_dir / "SOURCE.txt"
    meta.write_text(
        f"Copied from {RESNET_SPLITS}\nseed={SEED}\nDo not reshuffle.\n"
    )
    return out_dir


def write_bags_csv(out_path: Path | None = None) -> Path:
    """CLAM extract_features_fp bags list (slide_id column)."""
    out_path = out_path or (ROOT / "bags_list.csv")
    labels = load_labels()
    # Only slides that have patch coords.
    rows = []
    for sid, lab in labels[["slide_id", "label"]].itertuples(index=False):
        h5 = PATCHES / "patches" / f"{sid}.h5"
        if h5.is_file():
            rows.append({"slide_id": sid, "label": lab})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path


def fold_split_dfs(fold: int, splits_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    splits_dir = splits_dir or (ROOT / "splits")
    df = pd.read_csv(splits_dir / f"fold_{fold}.csv")
    return {
        "train": df[df["split"] == "train"].reset_index(drop=True),
        "val": df[df["split"] == "val"].reset_index(drop=True),
        "test": df[df["split"] == "test"].reset_index(drop=True),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sync-splits", action="store_true")
    p.add_argument("--bags-csv", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.sync_splits or (not args.bags_csv):
        d = sync_splits()
        print(f"splits → {d}")
    if args.bags_csv or (not args.sync_splits):
        p = write_bags_csv()
        print(f"bags → {p} n={len(pd.read_csv(p))}")


if __name__ == "__main__":
    main()
