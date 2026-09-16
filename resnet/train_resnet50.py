#!/usr/bin/env python3
"""Train & eval ResNet50 on 256×256 slide thumbnails (no MIL), 5-fold seeded CV."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SlideThumbDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, train: bool):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.image_dir / f"{row['slide_id']}.png"
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        y = int(row["label"])
        return x, y


def build_model(num_classes: int = 2) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_y, all_p, all_prob = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        all_y.append(y.numpy())
        all_p.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    y_prob = np.concatenate(all_prob)
    acc = float(accuracy_score(y_true, y_pred))
    # AUC undefined if only one class present in the split.
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    return {"acc": acc, "auc": auc, "n": int(len(y_true))}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


def write_epoch_txt(path: Path, fold: int, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"fold={fold}",
        f"epoch={epoch}",
        f"seed={SEED}",
    ]
    for split in ("train", "val", "test"):
        m = metrics[split]
        lines.append(
            f"{split}: acc={m['acc']:.6f} auc={m['auc']:.6f} n={m['n']}"
        )
    path.write_text("\n".join(lines) + "\n")


def run_fold(
    fold: int,
    split_csv: Path,
    image_dir: Path,
    result_dir: Path,
    ckpt_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    device: torch.device,
) -> dict:
    df = pd.read_csv(split_csv)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_loader = DataLoader(
        SlideThumbDataset(train_df, image_dir, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        SlideThumbDataset(val_df, image_dir, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        SlideThumbDataset(test_df, image_dir, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    # Separate loader for train metrics without augmentation.
    train_eval_loader = DataLoader(
        SlideThumbDataset(train_df, image_dir, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    set_seed(SEED + fold)
    model = build_model().to(device)
    # Class imbalance: ~94 neg / 36 pos
    n_neg = int((train_df["label"] == 0).sum())
    n_pos = int((train_df["label"] == 1).sum())
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    fold_result_dir = result_dir / f"fold_{fold}"
    best_val_auc = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        metrics = {
            "train": evaluate(model, train_eval_loader, device),
            "val": evaluate(model, val_loader, device),
            "test": evaluate(model, test_loader, device),
        }
        write_epoch_txt(
            fold_result_dir / f"epoch_{epoch:03d}.txt",
            fold,
            epoch,
            metrics,
        )
        row = {
            "fold": fold,
            "epoch": epoch,
            "loss": loss,
            "train_acc": metrics["train"]["acc"],
            "train_auc": metrics["train"]["auc"],
            "val_acc": metrics["val"]["acc"],
            "val_auc": metrics["val"]["auc"],
            "test_acc": metrics["test"]["acc"],
            "test_auc": metrics["test"]["auc"],
        }
        history.append(row)
        print(
            f"[fold {fold}] epoch {epoch:03d} loss={loss:.4f} "
            f"train acc={row['train_acc']:.3f} auc={row['train_auc']:.3f} | "
            f"val acc={row['val_acc']:.3f} auc={row['val_auc']:.3f} | "
            f"test acc={row['test_acc']:.3f} auc={row['test_auc']:.3f}"
        )

        val_auc = metrics["val"]["auc"]
        if not np.isnan(val_auc) and val_auc >= best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_auc": val_auc,
                    "seed": SEED,
                },
                ckpt_dir / f"fold_{fold}_best.pt",
            )

    # Final summary for this fold (best by val AUC).
    best_row = next(r for r in history if r["epoch"] == best_epoch)
    summary = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_auc": best_row["val_auc"],
        "best_val_acc": best_row["val_acc"],
        "test_auc_at_best_val": best_row["test_auc"],
        "test_acc_at_best_val": best_row["test_acc"],
    }
    (fold_result_dir / "fold_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(history).to_csv(fold_result_dir / "history.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", type=Path, default=root / "cache_256")
    p.add_argument("--splits-dir", type=Path, default=root / "resnet50" / "splits")
    p.add_argument("--result-dir", type=Path, default=root / "resnet50" / "result")
    p.add_argument("--ckpt-dir", type=Path, default=root / "resnet50" / "checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--folds", type=str, default="0,1,2,3,4")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global SEED
    SEED = args.seed
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seed={SEED}")

    fold_ids = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    summaries = []
    for fold in fold_ids:
        split_csv = args.splits_dir / f"fold_{fold}.csv"
        if not split_csv.is_file():
            raise FileNotFoundError(split_csv)
        summaries.append(
            run_fold(
                fold,
                split_csv,
                args.image_dir,
                args.result_dir,
                args.ckpt_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=args.num_workers,
                device=device,
            )
        )

    out = args.result_dir / "cv_summary.json"
    out.write_text(json.dumps(summaries, indent=2))
    # Mean ± std over folds (test at best val).
    test_aucs = [s["test_auc_at_best_val"] for s in summaries]
    test_accs = [s["test_acc_at_best_val"] for s in summaries]
    lines = [
        f"seed={SEED}",
        f"n_folds={len(summaries)}",
        f"test_acc_mean={np.mean(test_accs):.6f}",
        f"test_acc_std={np.std(test_accs):.6f}",
        f"test_auc_mean={np.mean(test_aucs):.6f}",
        f"test_auc_std={np.std(test_aucs):.6f}",
    ]
    (args.result_dir / "cv_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
