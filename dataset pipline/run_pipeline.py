#!/usr/bin/env python3
"""End-to-end SLN-Breast MIL dataset pipeline.

Steps:
  1) Download WSIs + target.csv from Hugging Face
  2) Clone CLAM and install strict-white preset
  3) Segment tissue and extract 256×256 patch coordinates

Default paths (under workspace):
  data/sln-breast/          raw SVS + target.csv  (single copy)
  mil-set/CLAM/             CLAM code
  mil-set/patches_256/      masks / patches(.h5 coords) / stitches

Example:
  python "dataset pipline/run_pipeline.py"
  python "dataset pipline/run_pipeline.py" --skip-download --mask-example-only
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PIPE_DIR = Path(__file__).resolve().parent
WORKSPACE = PIPE_DIR.parents[0]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--wsi-dir",
        type=Path,
        default=WORKSPACE / "data" / "sln-breast",
        help="Raw WSI output/input directory (no second copy is made).",
    )
    p.add_argument(
        "--patch-dir",
        type=Path,
        default=WORKSPACE / "mil-set" / "patches_256",
    )
    p.add_argument(
        "--clam-dir",
        type=Path,
        default=WORKSPACE / "mil-set" / "CLAM",
    )
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--step-size", type=int, default=256)
    p.add_argument("--labels-only", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-setup", action="store_true")
    p.add_argument("--skip-patch", action="store_true")
    p.add_argument("--force-clam", action="store_true")
    p.add_argument(
        "--mask-example-only",
        action="store_true",
        help="Only segment one slide for mask QC; do not run full patching.",
    )
    p.add_argument("--example-slide", default=None)
    p.add_argument("--no-stitch", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = PIPE_DIR / "configs" / "sln_strict_white.csv"

    print("=== SLN-Breast MIL pipeline ===")
    print(f"workspace : {WORKSPACE}")
    print(f"wsi_dir   : {args.wsi_dir}")
    print(f"patch_dir : {args.patch_dir}")
    print(f"clam_dir  : {args.clam_dir}")
    print(f"preset    : {preset} (sthresh=1 → only near-white removed)")

    if not args.skip_download:
        dl = _load("sln_download", PIPE_DIR / "01_download.py")
        dl.download_sln_breast(args.wsi_dir, labels_only=args.labels_only)
    else:
        print("[download] skipped")

    if args.labels_only:
        print("[pipeline] --labels-only set; stopping after download.")
        return

    if not args.skip_setup:
        setup = _load("sln_setup", PIPE_DIR / "02_setup_clam.py")
        setup.setup_clam(args.clam_dir, preset, force=args.force_clam)
    else:
        print("[setup] skipped")

    if not args.skip_patch:
        patch_dir = args.patch_dir
        # Keep QC masks out of the full run directory unless user overrode --patch-dir.
        default_full = WORKSPACE / "mil-set" / "patches_256"
        if args.mask_example_only and patch_dir.resolve() == default_full.resolve():
            patch_dir = WORKSPACE / "mil-set" / "patches_256_example"
            print(f"[patch] mask example → {patch_dir}")

        patch = _load("sln_patch", PIPE_DIR / "03_create_patches.py")
        patch.create_patches(
            source_dir=args.wsi_dir,
            save_dir=patch_dir,
            clam_dir=args.clam_dir,
            preset_src=preset,
            patch_size=args.patch_size,
            step_size=args.step_size,
            seg=True,
            patch=not args.mask_example_only,
            stitch=(not args.mask_example_only) and (not args.no_stitch),
            mask_example_only=args.mask_example_only,
            example_slide=args.example_slide,
        )
    else:
        print("[patch] skipped")

    print("=== pipeline finished ===")


if __name__ == "__main__":
    main()
