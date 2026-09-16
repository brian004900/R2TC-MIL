#!/usr/bin/env python3
"""ACMIL on SLN using official architecture.transformer.ACMIL_GA.

Embed: ImageNet ResNet50-trunc 1024-d in this repo's feats/.
Splits: exact /resnet 5-fold.
Default ACMIL: n_token=5, n_masked_patch=10, mask_drop=0.6
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[0] / "_common"))

from architecture.transformer import ACMIL_GA  # noqa: E402
from mil_utils import (  # noqa: E402
    SEED,
    BagFeatDataset,
    load_fold,
    metrics_from_scores,
    save_cv_summary,
    set_seed,
    write_epoch_txt,
)


def make_conf(n_token=5):
    return types.SimpleNamespace(
        D_feat=1024,
        D_inner=512,
        n_class=2,
        n_token=n_token,
    )


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for feats, y, _ in loader:
        bag = feats.to(device)  # [1,N,D] from collate? batch=1 → [1,N,D] if unsqueezed
        if bag.dim() == 2:
            bag = bag.unsqueeze(0)
        # DataLoader batch_size=1 yields feats [1,N,D] already if dataset returns [N,D]
        x = feats.squeeze(0)
        sub, slide_logit, _ = model([x.to(device)])
        prob = torch.softmax(slide_logit, dim=-1)[0, 1].item()
        ys.append(int(y.item()))
        ps.append(prob)
    return metrics_from_scores(ys, ps)


def train_fold(fold, feat_dir, result_dir, epochs, lr, device, n_token, n_masked, mask_drop):
    splits = load_fold(fold)
    loaders = {
        s: DataLoader(BagFeatDataset(splits[s], feat_dir, max_patches=4096, train=(s=="train")), batch_size=1, shuffle=(s == "train"))
        for s in ("train", "val", "test")
    }
    set_seed(SEED + fold)
    conf = make_conf(n_token)
    model = ACMIL_GA(
        conf, n_token=n_token, n_masked_patch=n_masked, mask_drop=mask_drop
    ).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, best_ep, history = -1.0, -1, []
    fold_dir = result_dir / f"fold_{fold}"
    for epoch in range(1, epochs + 1):
        model.train()
        for feats, y, _ in loaders["train"]:
            x = feats.squeeze(0).to(device)
            label = torch.tensor([int(y.item())], device=device)
            opt.zero_grad(set_to_none=True)
            sub_preds, slide_preds, _ = model([x])
            # slide_preds: [1, n_class] or [n_class]; sub_preds: [n_token, n_class]
            if slide_preds.dim() == 1:
                slide_preds = slide_preds.unsqueeze(0)
            loss = crit(slide_preds, label)
            if n_token > 1:
                if sub_preds.dim() == 3:
                    sub_preds = sub_preds.squeeze(1)
                loss = loss + crit(sub_preds, label.repeat(n_token))
            loss.backward()
            opt.step()
        metrics = {
            "train": eval_loader(model, loaders["train"], device),
            "val": eval_loader(model, loaders["val"], device),
        }
        write_epoch_txt(fold_dir / f"epoch_{epoch:03d}.txt", "ACMIL", fold, epoch, metrics)
        print(
            f"[ACMIL fold{fold}] ep{epoch:03d} "
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
        f"[ACMIL fold{fold}] best_ep={best_ep} "
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
    p.add_argument("--n-token", type=int, default=5)
    p.add_argument("--n-masked-patch", type=int, default=10)
    p.add_argument("--mask-drop", type=float, default=0.6)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = [
        train_fold(
            int(f),
            args.feat_dir,
            args.result_dir,
            args.epochs,
            args.lr,
            device,
            args.n_token,
            args.n_masked_patch,
            args.mask_drop,
        )
        for f in args.folds.split(",")
    ]
    save_cv_summary(args.result_dir, summaries, "ACMIL")


if __name__ == "__main__":
    main()
