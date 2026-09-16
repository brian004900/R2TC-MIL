#!/usr/bin/env python3
"""Step 3: CLAM tissue seg + 256 patch coordinates (strict white-only background).

Uses the original SVS directory in-place (no source copy / symlink tree).
Non-.svs files such as target.csv are excluded via a process list.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

PRESET_NAME = "sln_strict_white.csv"
PROCESS_LIST_NAME = "process_list_sln.csv"

# Strict: only drop near-white background (low HSV saturation).
DEFAULT_PARAMS = {
    "seg_level": -1,
    "sthresh": 1,
    "mthresh": 7,
    "close": 2,
    "use_otsu": False,
    "keep_ids": "none",
    "exclude_ids": "none",
    "a_t": 1.0,
    "a_h": 1.0,
    "max_n_holes": 8,
    "vis_level": -1,
    "line_thickness": 250,
    "use_padding": True,
    "contour_fn": "four_pt",
}


def list_svs(source_dir: Path) -> list[str]:
    return sorted(p.name for p in source_dir.glob("*.svs"))


def write_process_list(
    save_dir: Path,
    slides: list[str],
    *,
    example_slide: str | None = None,
) -> Path:
    """Write CLAM process_list under save_dir; only selected slides have process=1."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / PROCESS_LIST_NAME
    fieldnames = [
        "slide_id",
        "process",
        "status",
        *DEFAULT_PARAMS.keys(),
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in slides:
            process = 1
            if example_slide is not None:
                process = 1 if name == example_slide else 0
            row = {
                "slide_id": name,
                "process": process,
                "status": "tbp",
                **DEFAULT_PARAMS,
            }
            writer.writerow(row)
    return path


def create_patches(
    *,
    source_dir: Path,
    save_dir: Path,
    clam_dir: Path,
    preset_src: Path,
    patch_size: int = 256,
    step_size: int = 256,
    seg: bool = True,
    patch: bool = True,
    stitch: bool = True,
    mask_example_only: bool = False,
    example_slide: str | None = None,
) -> None:
    source_dir = source_dir.resolve()
    save_dir = save_dir.resolve()
    clam_dir = clam_dir.resolve()

    if not clam_dir.is_dir():
        raise FileNotFoundError(f"CLAM not found: {clam_dir} (run 02_setup_clam.py)")

    slides = list_svs(source_dir)
    if not slides:
        raise FileNotFoundError(f"No .svs under {source_dir}")

    if mask_example_only:
        example_slide = example_slide or slides[0]
        if example_slide not in slides:
            raise FileNotFoundError(f"example slide missing: {example_slide}")
        patch = False
        stitch = False
        print(f"[patch] mask-example-only: {example_slide}")

    # Keep preset in sync with pipeline config (no dependency on prior edits).
    dest_preset = clam_dir / "presets" / PRESET_NAME
    shutil.copy2(preset_src, dest_preset)

    process_list = write_process_list(
        save_dir, slides, example_slide=example_slide if mask_example_only else None
    )
    print(f"[patch] source={source_dir} ({len(slides)} svs)")
    print(f"[patch] save_dir={save_dir}")
    print(f"[patch] process_list={process_list}")
    print(
        f"[patch] size={patch_size} step={step_size} "
        f"seg={seg} patch={patch} stitch={stitch}"
    )
    print(f"[patch] preset={dest_preset} (sthresh=1, a_t=1, a_h=1)")

    cmd = [
        sys.executable,
        "create_patches_fp.py",
        "--source",
        str(source_dir),
        "--save_dir",
        str(save_dir),
        "--patch_size",
        str(patch_size),
        "--step_size",
        str(step_size),
        "--preset",
        PRESET_NAME,
        "--process_list",
        PROCESS_LIST_NAME,
    ]
    if seg:
        cmd.append("--seg")
    if patch:
        cmd.append("--patch")
    if stitch:
        cmd.append("--stitch")

    subprocess.run(cmd, cwd=str(clam_dir), check=True)
    print("[patch] done")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    workspace = root.parents[0]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir",
        type=Path,
        default=workspace / "data" / "sln-breast",
        help="Original WSI directory (used in-place).",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=workspace / "mil-set" / "patches_256",
    )
    p.add_argument(
        "--clam-dir",
        type=Path,
        default=workspace / "mil-set" / "CLAM",
    )
    p.add_argument(
        "--preset",
        type=Path,
        default=root / "configs" / PRESET_NAME,
    )
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--step-size", type=int, default=256)
    p.add_argument("--no-seg", action="store_true")
    p.add_argument("--no-patch", action="store_true")
    p.add_argument("--no-stitch", action="store_true")
    p.add_argument(
        "--mask-example-only",
        action="store_true",
        help="Segment one slide only (for mask QC); skip patch/stitch.",
    )
    p.add_argument("--example-slide", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    create_patches(
        source_dir=args.source_dir,
        save_dir=args.save_dir,
        clam_dir=args.clam_dir,
        preset_src=args.preset,
        patch_size=args.patch_size,
        step_size=args.step_size,
        seg=not args.no_seg,
        patch=not args.no_patch,
        stitch=not args.no_stitch,
        mask_example_only=args.mask_example_only,
        example_slide=args.example_slide,
    )


if __name__ == "__main__":
    main()
