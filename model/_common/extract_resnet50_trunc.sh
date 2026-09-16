#!/usr/bin/env bash
# Extract ImageNet ResNet50-trunc (1024-d) features via CLAM — no SLN fine-tuning (no leakage).
# Usage: bash extract_resnet50_trunc.sh /path/to/out_feat_dir
set -euo pipefail
OUT="${1:?feat dir}"
WORKSPACE="$(cd "$(dirname "$0")/../.." && pwd)"
CLAM="$WORKSPACE/mil-set/CLAM"
CSV="$WORKSPACE/model/_common/bags_list.csv"
H5="$WORKSPACE/mil-set/patches_256"
SLIDE="$WORKSPACE/data/sln-breast"
mkdir -p "$OUT"
cd "$CLAM"
python3 extract_features_fp.py \
  --data_h5_dir "$H5" \
  --data_slide_dir "$SLIDE" \
  --csv_path "$CSV" \
  --feat_dir "$OUT" \
  --slide_ext .svs \
  --model_name resnet50_trunc \
  --batch_size 256 \
  --target_patch_size 224
echo "features → $OUT/pt_files"
