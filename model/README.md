# MIL Models on SLN-Breast (aligned with `/resnet` 5-fold)

Cloned / adapted repos (separate modules):

| Dir | Upstream / note |
|-----|----------|
| `ABMIL/` | https://github.com/AMLab-Amsterdam/AttentionDeepMIL |
| `DSMIL/` | https://github.com/binli123/dsmil-wsi |
| `DTFD-MIL/` | https://github.com/hrzhang1123/DTFD-MIL |
| `ACMIL/` | https://github.com/dazhangyu123/ACMIL |
| `R2T-MIL/` | https://github.com/DearCaat/RRT-MIL |
| `R2TC-MIL/` | R2T-MIL + Cluster-conditioned Instance Reweighting

## No leakage / fair splits

- Train/val/test IDs are **copied from** `/workspace/resnet/resnet50/splits` (`seed=42`).
- Feature backbone is **ImageNet pretrained ResNet50-trunc (1024-d)** via CLAM; **not** trained on SLN labels.
- Each method stores features under its own `feats/pt_files/` and trains with `run_sln_5fold.py` separately.

## Run one method

```bash
python3 model/_common/prepare_splits.py --sync-splits --bags-csv

cd model/ABMIL && bash ../_common/embed_for_method.sh . && python3 run_sln_5fold.py --epochs 50
cd ../AE-MIL && bash ../_common/embed_for_method.sh . && python3 run_sln_5fold.py --epochs 50
```

Results: `<method>/results/fold_k/epoch_XXX.txt` (train/val only) and `cv_summary.txt` (test acc & auc at **first-best val AUC**).
