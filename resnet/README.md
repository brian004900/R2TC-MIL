# ResNet50 slide-level baseline (no MIL)

WSI → resize to **256×256** → ImageNet-pretrained **ResNet50** binary classification.  
Fixed **seed=42** stratified **5-fold**, with each fold **train/val/test ≈ 60/20/20**, and class ratios in each split close to the overall balance (94:36).

## Quick start

```bash
cd /workspace/resnet
python prepare_cache256.py          # cache 256 PNGs (once)
python make_splits.py               # write resnet50/splits/fold_*.csv
python train_resnet50.py --epochs 30
```

## Outputs

| Path | Content |
|------|------|
| `cache_256/*.png` | 256×256 image per slide |
| `resnet50/splits/fold_{0-4}.csv` | reproducible splits |
| `resnet50/result/fold_k/epoch_XXX.txt` | per-epoch train/val/test **acc & auc** |
| `resnet50/result/cv_summary.txt` | 5-fold summary |
| `resnet50/checkpoints/fold_k_best.pt` | best weights by val AUC |

Example `epoch_XXX.txt`:

```
fold=0
epoch=1
seed=42
train: acc=0.7 auc=0.7 n=78
val: acc=0.6 auc=0.6 n=26
test: acc=0.6 auc=0.6 n=26
```

## Fair comparison

- `SEED = 42` (`make_splits.py` / `train_resnet50.py`)
- The same `resnet50/splits/` can be reused by downstream MIL methods
