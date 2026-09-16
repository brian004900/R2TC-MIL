# SLN-Breast MIL Dataset Pipeline

Download SLN-Breast from Hugging Face, run CLAM segmentation, and produce **256×256** patch coordinates to reproduce the MIL dataset used in this repo.

## One-shot run

```bash
# System dependency (OpenSlide)
sudo apt-get install -y openslide-tools

pip install -r "dataset pipline/requirements.txt"
python "dataset pipline/run_pipeline.py"
```

Skip steps when data/patches already exist:

```bash
python "dataset pipline/run_pipeline.py" --skip-download
python "dataset pipline/run_pipeline.py" --mask-example-only   # preview one mask first
python "dataset pipline/run_pipeline.py" --skip-download --skip-setup
```

Or run step by step: `01_download.py` → `02_setup_clam.py` → `03_create_patches.py`.

## Directory layout (no WSI copies)

| Path | Content |
|------|------|
| `data/sln-breast/` | original `.svs` + `target.csv` (single copy) |
| `mil-set/CLAM/` | [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM) |
| `mil-set/patches_256/masks/` | segmentation visualizations |
| `mil-set/patches_256/patches/` | one `.h5` per slide (patch top-left coords) |
| `mil-set/patches_256/stitches/` | optional stitch previews |

WSIs are read **in place**. A process list restricts processing to `.svs` files, so no extra source copy is created.

## Segmentation settings (strict white-only)

Config: `configs/sln_strict_white.csv`

| Param | Value | Notes |
|------|-----|------|
| `sthresh` | **1** | HSV saturation threshold; default is 8. Lower values remove near-white background only |
| `mthresh` | 7 | median-filter kernel |
| `close` | 2 | light morphological closing |
| `use_otsu` | False | do not use Otsu |
| `a_t` | 1 | keep very small tissue contours |
| `a_h` | 1 | also exclude relatively small holes |
| `max_n_holes` | 8 | max holes processed per contour |
| `contour_fn` | four_pt | whether a patch falls inside tissue |
| `patch_size` / `step_size` | 256 / 256 | non-overlapping 256 patches |

Data source: `Refrainkana33/sln-breast-tcia-svs` (TCIA SLN-Breast). In `target.csv`, `target=1/0` are slide-level (bag) labels.

## Output notes

Each `patches/<slide_id>.h5` contains `coords` of shape `(N, 2)`: the level-0 top-left `(x, y)` of every 256×256 patch, for downstream feature extraction (e.g. CLAM `extract_features_fp.py`).
