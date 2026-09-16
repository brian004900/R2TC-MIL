#!/usr/bin/env bash
# Usage: bash embed_for_method.sh /path/to/METHOD_DIR
set -euo pipefail
METHOD_DIR="$(cd "${1:?method dir}" && pwd)"
BASE="$(cd "$(dirname "$0")/.." && pwd)"
COMMON="$(cd "$(dirname "$0")" && pwd)"
OUT="$METHOD_DIR/feats"

SRC_CANDIDATES=(
  "$COMMON/feats_resnet50_trunc"
  "$BASE/ABMIL/feats"
  "$BASE/DSMIL/feats"
  "$BASE/DTFD-MIL/feats"
  "$BASE/MHIM-MIL/feats"
  "$BASE/ACMIL/feats"
  "$BASE/R2T-MIL/feats"
  "$BASE/R2TC-MIL/feats"
  "$BASE/AE-MIL/feats"
)

mkdir -p "$OUT/pt_files"
if [[ "$(ls -A "$OUT/pt_files" 2>/dev/null | wc -l)" -ge 130 ]]; then
  echo "[embed] $METHOD_DIR already has features"
  exit 0
fi

for src in "${SRC_CANDIDATES[@]}"; do
  if [[ -d "$src/pt_files" ]] && [[ "$(ls -A "$src/pt_files" 2>/dev/null | wc -l)" -ge 130 ]]; then
    echo "[embed] linking from $src → $OUT"
    cp -al "$src/pt_files/." "$OUT/pt_files/" 2>/dev/null || cp -a "$src/pt_files/." "$OUT/pt_files/"
    echo "[embed] $(ls "$OUT/pt_files" | wc -l) files"
    exit 0
  fi
done

echo "[embed] extracting ResNet50-trunc into $OUT"
bash "$COMMON/extract_resnet50_trunc.sh" "$OUT"
