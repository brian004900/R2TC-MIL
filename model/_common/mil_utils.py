#!/usr/bin/env python3
"""Shared MIL bag dataset + metrics for SLN 5-fold (aligned with /resnet splits)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import Dataset

SEED = 42
COMMON = Path(__file__).resolve().parent
WORKSPACE = COMMON.parents[1]
SPLITS_DIR = COMMON / "splits"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BagFeatDataset(Dataset):
    """Load precomputed bag features (.pt as [N,D] or .npy)."""

    def __init__(self, df: pd.DataFrame, feat_dir: Path, max_patches: int | None = None, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.feat_dir = Path(feat_dir)
        self.max_patches = max_patches
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sid = str(row["slide_id"])
        pt = self.feat_dir / f"{sid}.pt"
        npy = self.feat_dir / f"{sid}.npy"
        if pt.is_file():
            feats = torch.load(pt, map_location="cpu", weights_only=True)
            if isinstance(feats, dict):
                feats = feats["features"]
            if not torch.is_tensor(feats):
                feats = torch.from_numpy(np.asarray(feats))
        elif npy.is_file():
            feats = torch.from_numpy(np.load(npy))
        else:
            raise FileNotFoundError(f"missing features for {sid} under {self.feat_dir}")
        feats = feats.float()
        if self.max_patches is not None and feats.shape[0] > self.max_patches:
            if self.train:
                idx_keep = torch.randperm(feats.shape[0])[: self.max_patches]
            else:
                # Deterministic subsample for val/test (no leakage, reproducible).
                g = torch.Generator()
                g.manual_seed(abs(hash(sid)) % (2**31))
                idx_keep = torch.randperm(feats.shape[0], generator=g)[: self.max_patches]
            feats = feats[idx_keep]
        label = int(row["label"])
        return feats, label, sid


def load_fold(fold: int, splits_dir: Path = SPLITS_DIR) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(Path(splits_dir) / f"fold_{fold}.csv")
    # Ensure only train/val/test rows for this fold are used (no leakage).
    out = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].copy().reset_index(drop=True)
        assert set(part["fold"].unique()) <= {fold}
        out[split] = part
    # Disjointness check
    ids = {s: set(out[s]["slide_id"]) for s in out}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    return out


def metrics_from_scores(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    return {"acc": acc, "auc": auc, "n": int(len(y_true))}


def write_epoch_txt(path: Path, method: str, fold: int, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"method={method}", f"fold={fold}", f"epoch={epoch}", f"seed={SEED}"]
    for split in ("train", "val", "test"):
        if split not in metrics:
            continue
        m = metrics[split]
        lines.append(f"{split}: acc={m['acc']:.6f} auc={m['auc']:.6f} n={m['n']}")
    path.write_text("\n".join(lines) + "\n")


def save_cv_summary(result_dir: Path, summaries: list[dict], method: str) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "cv_summary.json").write_text(json.dumps(summaries, indent=2))
    test_aucs = [s["test_auc_at_best_val"] for s in summaries]
    test_accs = [s["test_acc_at_best_val"] for s in summaries]
    lines = [
        f"method={method}",
        f"seed={SEED}",
        f"n_folds={len(summaries)}",
        f"test_acc_mean={np.mean(test_accs):.6f}",
        f"test_acc_std={np.std(test_accs):.6f}",
        f"test_auc_mean={np.mean(test_aucs):.6f}",
        f"test_auc_std={np.std(test_aucs):.6f}",
        "splits=aligned_with_/resnet/resnet50/splits",
    ]
    (result_dir / "cv_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
