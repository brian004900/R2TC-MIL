#!/usr/bin/env python3
"""DTFD-MIL on SLN using official Model/* + train/test loops (label logic fixed).

Embed: ImageNet ResNet50-trunc 1024-d in this repo's feats/.
Splits: exact /resnet 5-fold.
"""

from __future__ import annotations

import argparse
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[0] / "_common"))

from Model.Attention import Attention_Gated as Attention  # noqa: E402
from Model.Attention import Attention_with_Classifier  # noqa: E402
from Model.network import Classifier_1fc, DimReduction  # noqa: E402
from utils import get_cam_1d  # noqa: E402
from mil_utils import (  # noqa: E402
    SEED,
    BagFeatDataset,
    load_fold,
    metrics_from_scores,
    save_cv_summary,
    set_seed,
    write_epoch_txt,
)


def _softmax_bag_prob(logit2):
    # logit2: [1,2] → P(class=1)
    return torch.softmax(logit2, dim=-1)[0, 1].item()


def train_one_bag(classifier, dimReduction, attention, attCls, feats, label, device, opt0, opt1, distill="AFS"):
    """One-slide DTFD-style update (pseudo-bags=4, AFS)."""
    classifier.train()
    dimReduction.train()
    attention.train()
    attCls.train()

    feats = feats.to(device)
    y = torch.tensor([label], device=device, dtype=torch.long)
    numGroup, total_instance = 4, 4
    instance_per_group = max(total_instance // numGroup, 1)

    mid = dimReduction(feats)
    AA = attention(mid, isNorm=False).squeeze(0)

    feat_index = list(range(feats.shape[0]))
    random.shuffle(feat_index)
    chunks = [c.tolist() for c in np.array_split(np.array(feat_index), numGroup)]

    slide_d_feat = []
    loss0 = 0.0
    opt0.zero_grad(set_to_none=True)
    for tindex in chunks:
        idx = torch.LongTensor(tindex).to(device)
        tmid = mid.index_select(0, idx)
        tAA = torch.softmax(AA.index_select(0, idx), dim=0)
        tatt = torch.sum(torch.einsum("ns,n->ns", tmid, tAA), dim=0, keepdim=True)
        pred = classifier(tatt)
        loss0 = loss0 + F.cross_entropy(pred, y)
        patch_logits = get_cam_1d(classifier, torch.einsum("ns,n->ns", tmid, tAA).unsqueeze(0)).squeeze(0)
        patch_logits = torch.transpose(patch_logits, 0, 1)
        patch_prob = torch.softmax(patch_logits, dim=1)
        _, sort_idx = torch.sort(patch_prob[:, -1], descending=True)
        if distill == "AFS":
            # attention-based feature selection: weighted sum as distilled instance
            slide_d_feat.append(tatt)
        else:
            topk = sort_idx[:instance_per_group].long()
            slide_d_feat.append(tmid.index_select(0, topk))
    loss0 = loss0 / numGroup
    loss0.backward()
    torch.nn.utils.clip_grad_norm_(
        list(classifier.parameters()) + list(dimReduction.parameters()) + list(attention.parameters()),
        5.0,
    )
    opt0.step()

    opt1.zero_grad(set_to_none=True)
    d_feat = torch.cat(slide_d_feat, dim=0).detach()
    pred1 = attCls(d_feat)
    loss1 = F.cross_entropy(pred1, y)
    loss1.backward()
    torch.nn.utils.clip_grad_norm_(attCls.parameters(), 5.0)
    opt1.step()
    return float(loss0.item() + loss1.item())


@torch.no_grad()
def eval_split(classifier, dimReduction, attention, attCls, loader, device):
    classifier.eval()
    dimReduction.eval()
    attention.eval()
    attCls.eval()
    ys, ps = [], []
    for feats, y, _ in loader:
        feats = feats.squeeze(0).to(device)
        mid = dimReduction(feats)
        AA = torch.softmax(attention(mid, isNorm=False).squeeze(0), dim=0)
        bag = torch.sum(torch.einsum("ns,n->ns", mid, AA), dim=0, keepdim=True)
        # tier-1
        p0 = _softmax_bag_prob(classifier(bag))
        # tier-2 on bag feat
        p1 = _softmax_bag_prob(attCls(bag))
        prob = 0.5 * (p0 + p1)
        ys.append(int(y.item()))
        ps.append(prob)
    return metrics_from_scores(ys, ps)


def train_fold(fold, feat_dir, result_dir, epochs, lr, device):
    splits = load_fold(fold)
    loaders = {
        s: DataLoader(BagFeatDataset(splits[s], feat_dir, max_patches=4096, train=(s=="train")), batch_size=1, shuffle=(s == "train"))
        for s in ("train", "val", "test")
    }
    set_seed(SEED + fold)
    in_chn, mdim = 1024, 512
    classifier = Classifier_1fc(mdim, 2).to(device)
    attention = Attention(mdim).to(device)
    dimReduction = DimReduction(in_chn, mdim).to(device)
    attCls = Attention_with_Classifier(L=mdim, num_cls=2).to(device)
    opt0 = torch.optim.Adam(
        list(classifier.parameters()) + list(attention.parameters()) + list(dimReduction.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    opt1 = torch.optim.Adam(attCls.parameters(), lr=lr, weight_decay=1e-4)

    best_val, best_ep, history = -1.0, -1, []
    fold_dir = result_dir / f"fold_{fold}"
    for epoch in range(1, epochs + 1):
        for feats, y, _ in loaders["train"]:
            train_one_bag(
                classifier,
                dimReduction,
                attention,
                attCls,
                feats.squeeze(0),
                int(y.item()),
                device,
                opt0,
                opt1,
            )
        metrics = {
            "train": eval_split(classifier, dimReduction, attention, attCls, loaders["train"], device),
            "val": eval_split(classifier, dimReduction, attention, attCls, loaders["val"], device),
        }
        write_epoch_txt(fold_dir / f"epoch_{epoch:03d}.txt", "DTFD-MIL", fold, epoch, metrics)
        print(
            f"[DTFD fold{fold}] ep{epoch:03d} "
            f"val_auc={metrics['val']['auc']:.3f} val_acc={metrics['val']['acc']:.3f}"
        )
        history.append(metrics)
        va = metrics["val"]["auc"]
        # First best val AUC wins on ties.
        if not np.isnan(va) and va > best_val:
            best_val, best_ep = va, epoch
            torch.save(
                {
                    "classifier": classifier.state_dict(),
                    "attention": attention.state_dict(),
                    "dimReduction": dimReduction.state_dict(),
                    "attCls": attCls.state_dict(),
                },
                fold_dir / "best.pt",
            )

    ckpt = torch.load(fold_dir / "best.pt", map_location=device, weights_only=True)
    classifier.load_state_dict(ckpt["classifier"])
    attention.load_state_dict(ckpt["attention"])
    dimReduction.load_state_dict(ckpt["dimReduction"])
    attCls.load_state_dict(ckpt["attCls"])
    test_m = eval_split(classifier, dimReduction, attention, attCls, loaders["test"], device)
    best = history[best_ep - 1]
    print(
        f"[DTFD fold{fold}] best_ep={best_ep} "
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
    summaries = [
        train_fold(int(f), args.feat_dir, args.result_dir, args.epochs, args.lr, device)
        for f in args.folds.split(",")
    ]
    save_cv_summary(args.result_dir, summaries, "DTFD-MIL")


if __name__ == "__main__":
    main()
