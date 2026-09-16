#!/usr/bin/env python3
"""Cache each SLN-Breast WSI as a 256×256 RGB PNG (slide-level, no MIL)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def wsi_to_256(svs_path: Path, size: int = 256) -> Image.Image:
    slide = openslide.OpenSlide(str(svs_path))
    try:
        # Prefer native thumbnail; fall back to lowest-res level.
        thumb = slide.get_thumbnail((size * 4, size * 4)).convert("RGB")
    finally:
        slide.close()
    return thumb.resize((size, size), Image.Resampling.LANCZOS)


def cache_all(wsi_dir: Path, out_dir: Path, size: int = 256) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slides = sorted(wsi_dir.glob("*.svs"))
    if not slides:
        raise FileNotFoundError(f"No .svs in {wsi_dir}")
    for svs in tqdm(slides, desc=f"cache {size}x{size}"):
        out = out_dir / f"{svs.stem}.png"
        if out.is_file():
            continue
        img = wsi_to_256(svs, size=size)
        img.save(out)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wsi-dir",
        type=Path,
        default=root.parents[0] / "data" / "sln-breast",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "cache_256",
    )
    p.add_argument("--size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_all(args.wsi_dir, args.out_dir, size=args.size)
    n = len(list(args.out_dir.glob("*.png")))
    print(f"cached {n} images → {args.out_dir}")


if __name__ == "__main__":
    main()
