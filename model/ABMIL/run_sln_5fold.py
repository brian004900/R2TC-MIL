#!/usr/bin/env python3
"""ABMIL on SLN bags (feature-level Attention MIL from AttentionDeepMIL, adapted).

Embed: ImageNet ResNet50-trunc 1024-d (CLAM), frozen — no label leakage.
Splits: exact copy of /resnet 5-fold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[0] / "_common"))
from mil_utils import (  # noqa: E402
    SEED,
    BagFeatDataset,
    load_fold,
    metrics_from_scores,
    save_cv_summary,
    set_seed,
    write_epoch_txt,
)


class AttentionMIL(nn.Module):
    """Ilse et al. ABMIL on precomputed instance features (not raw MNIST pixels)."""

    def __init__(self, feats_size: int = 1024, M: int = 512, L: int = 128):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(feats_size, M), nn.ReLU())
        self.attention = nn.Sequential(
            nn.Linear(M, L),
            nn.Tanh(),
            nn.Linear(L, 1),
        )
        self.classifier = nn.Sequential(nn.Linear(M, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        # x: [N, D]
        h = self.feature(x)
        a = torch.transpose(self.attention(h), 1, 0)  # 1xN
        a = F.softmax(a, dim=1)
        z = torch.mm(a, h)  # 1xM
        p = self.classifier(z).view(-1)
        return p


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for feats, y, _ in loader:
        feats = feats.squeeze(0).to(device)
        p = model(feats).item()
        ys.append(int(y.item() if torch.is_tensor(y) else y))
        ps.append(float(p))
    return metrics_from_scores(ys, ps)


def train_fold(fold, feat_dir, result_dir, epochs, lr, device):
    splits = load_fold(fold)
    loaders = {
        s: DataLoader(
            BagFeatDataset(splits[s], feat_dir),
            batch_size=1,
            shuffle=(s == "train"),
            num_workers=0,
        )
        for s in ("train", "val", "test")
    }
    set_seed(SEED + fold)
    model = AttentionMIL().to(device)
    # imbalance weight via BCE pos_weight
    n_pos = int((splits["train"]["label"] == 1).sum())
    n_neg = int((splits["train"]["label"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # model already has sigmoid — use BCELoss
    crit = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, best_ep, history = -1.0, -1, []
    fold_dir = result_dir / f"fold_{fold}"
    for epoch in range(1, epochs + 1):
        model.train()
        for feats, y, _ in loaders["train"]:
            feats = feats.squeeze(0).to(device)
            y = torch.tensor([float(y.item())], device=device)
            opt.zero_grad(set_to_none=True)
            p = model(feats)
            loss = crit(p, y)
            loss.backward()
            opt.step()
        metrics = {
            "train": eval_loader(model, loaders["train"], device),
            "val": eval_loader(model, loaders["val"], device),
        }
        write_epoch_txt(fold_dir / f"epoch_{epoch:03d}.txt", "ABMIL", fold, epoch, metrics)
        print(
            f"[ABMIL fold{fold}] ep{epoch:03d} "
            f"val_auc={metrics['val']['auc']:.3f} val_acc={metrics['val']['acc']:.3f}"
        )
        history.append(metrics)
        va = metrics["val"]["auc"]
        # First best val AUC wins on ties.
        if not np.isnan(va) and va > best_val:
            best_val, best_ep = va, epoch
            torch.save(model.state_dict(), fold_dir / "best.pt")

    model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=device, weights_only=True))
    test_m = eval_loader(model, loaders["test"], device)
    best = history[best_ep - 1]
    print(
        f"[ABMIL fold{fold}] best_ep={best_ep} "
        f"val_auc={best['val']['auc']:.3f} test_auc={test_m['auc']:.3f} test_acc={test_m['acc']:.3f}"
    )
    return {
        "fold": fold,
        "best_epoch": best_ep,
        "best_val_auc": best["val"]["auc"],
        "best_val_acc": best["val"]["acc"],
        "test_auc_at_best_val": test_m["auc"],
        "test_acc_at_best_val": test_m["acc"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-dir", type=Path, default=ROOT / "feats" / "pt_files")
    p.add_argument("--result-dir", type=Path, default=ROOT / "results")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--folds", default="0,1,2,3,4")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = []
    for fold in [int(x) for x in args.folds.split(",")]:
        summaries.append(
            train_fold(fold, args.feat_dir, args.result_dir, args.epochs, args.lr, device)
        )
    save_cv_summary(args.result_dir, summaries, "ABMIL")


if __name__ == "__main__":
    main()
