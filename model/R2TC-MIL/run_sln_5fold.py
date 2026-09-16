#!/usr/bin/env python3
"""R2TC-MIL: RRTMIL + CRR (cluster re-weighting) on SLN.

Before bag classification, patches are k-means clustered (K=3), each cluster
goes through the shared classification head, and patches are re-weighted by
cluster logits then attention-pooled for the final prediction.

Embed: ImageNet ResNet50-trunc 1024-d (shared), frozen.
Splits: /resnet 5-fold (seed=42), no reshuffle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[0] / "_common"))

from modules.rrt import RRTMIL  # noqa: E402
from mil_utils import (  # noqa: E402
    SEED,
    BagFeatDataset,
    load_fold,
    metrics_from_scores,
    save_cv_summary,
    set_seed,
    write_epoch_txt,
)


def build_model(num_cluster: int = 3):
    return RRTMIL(
        input_dim=1024,
        n_classes=2,
        dropout=0.25,
        act="relu",
        pool="attn",
        da_act="tanh",
        n_layers=2,
        epeg_k=15,
        crmsa_k=1,
        crmsa_heads=8,
        all_shortcut=True,
        region_num=8,
        num_cluster=num_cluster,
        cluster_distance="euclidean",
        persistent_center=True,
        nor_index=0,
    )


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for feats, y, _ in loader:
        feats = feats.to(device)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        logits = model(feats)
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        ys.append(int(y.item() if torch.is_tensor(y) else y))
        ps.append(float(prob))
    return metrics_from_scores(ys, ps)


def train_fold(fold, feat_dir, result_dir, epochs, lr, device, max_patches, num_cluster):
    splits = load_fold(fold)
    loaders = {
        s: DataLoader(
            BagFeatDataset(
                splits[s],
                feat_dir,
                max_patches=max_patches,
                train=(s == "train"),
            ),
            batch_size=1,
            shuffle=(s == "train"),
            num_workers=0,
        )
        for s in ("train", "val", "test")
    }
    set_seed(SEED + fold)
    model = build_model(num_cluster=num_cluster).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, best_ep, history = -1.0, -1, []
    fold_dir = result_dir / f"fold_{fold}"
    for epoch in range(1, epochs + 1):
        model.train()
        for feats, y, _ in loaders["train"]:
            feats = feats.to(device)
            if feats.dim() == 2:
                feats = feats.unsqueeze(0)
            label = torch.tensor([int(y.item())], device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(feats)
            loss = crit(logits, label)
            loss.backward()
            opt.step()
        metrics = {
            "train": eval_loader(model, loaders["train"], device),
            "val": eval_loader(model, loaders["val"], device),
        }
        write_epoch_txt(fold_dir / f"epoch_{epoch:03d}.txt", "R2TC-MIL", fold, epoch, metrics)
        print(
            f"[R2TC-MIL fold{fold}] ep{epoch:03d} "
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
        f"[R2TC-MIL fold{fold}] best_ep={best_ep} "
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
    p.add_argument("--max-patches", type=int, default=4096)
    p.add_argument("--num-cluster", type=int, default=3)
    p.add_argument("--folds", default="0,1,2,3,4")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"R2TC-MIL device={device} lr={args.lr} max_patches={args.max_patches} "
        f"num_cluster={args.num_cluster}"
    )
    summaries = []
    for fold in [int(x) for x in args.folds.split(",")]:
        summaries.append(
            train_fold(
                fold,
                args.feat_dir,
                args.result_dir,
                args.epochs,
                args.lr,
                device,
                args.max_patches,
                args.num_cluster,
            )
        )
    save_cv_summary(args.result_dir, summaries, "R2TC-MIL")


if __name__ == "__main__":
    main()
