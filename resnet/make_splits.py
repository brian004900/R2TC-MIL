#!/usr/bin/env python3
"""Build seeded 5-fold train/val/test splits with preserved pos/neg ratio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# Fixed for fair future comparisons.
SEED = 42
N_FOLDS = 5
# After holding out test (~20%), val is 25% of the remainder → ~60/20/20.
VAL_FRACTION_OF_TRAINVAL = 0.25


def load_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"slide": "slide_id"})
    df["slide_id"] = df["slide_id"].astype(str).str.replace(".svs", "", regex=False)
    df["label"] = df["target"].astype(int)
    return df[["slide_id", "label"]].drop_duplicates().reset_index(drop=True)


def ratio_str(labels: np.ndarray) -> str:
    n = len(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return f"n={n} pos={n_pos} ({n_pos / n:.3f}) neg={n_neg} ({n_neg / n:.3f})"


def make_splits(df: pd.DataFrame, seed: int = SEED, n_folds: int = N_FOLDS) -> list[dict]:
    y = df["label"].to_numpy()
    indices = np.arange(len(df))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold, (tv_idx, te_idx) in enumerate(skf.split(indices, y)):
        y_tv = y[tv_idx]
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=VAL_FRACTION_OF_TRAINVAL,
            random_state=seed + fold,
        )
        tr_rel, va_rel = next(sss.split(tv_idx, y_tv))
        tr_idx = tv_idx[tr_rel]
        va_idx = tv_idx[va_rel]

        fold_info = {
            "fold": fold,
            "train": df.iloc[tr_idx].reset_index(drop=True),
            "val": df.iloc[va_idx].reset_index(drop=True),
            "test": df.iloc[te_idx].reset_index(drop=True),
        }
        folds.append(fold_info)
    return folds


def save_splits(folds: list[dict], out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": seed, "n_folds": len(folds), "folds": []}

    for item in folds:
        fold = item["fold"]
        rows = []
        for split_name in ("train", "val", "test"):
            part = item[split_name].copy()
            part["split"] = split_name
            part["fold"] = fold
            rows.append(part)
            summary["folds"].append(
                {
                    "fold": fold,
                    "split": split_name,
                    "stats": ratio_str(part["label"].to_numpy()),
                }
            )
        pd.concat(rows, ignore_index=True).to_csv(
            out_dir / f"fold_{fold}.csv", index=False
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--labels",
        type=Path,
        default=root.parents[0] / "data" / "sln-breast" / "target.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "resnet50" / "splits",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_labels(args.labels)
    print("overall:", ratio_str(df["label"].to_numpy()))
    folds = make_splits(df, seed=args.seed, n_folds=args.n_folds)
    save_splits(folds, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
