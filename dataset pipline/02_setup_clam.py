#!/usr/bin/env python3
"""Step 2: clone mahmoodlab/CLAM (shallow) and install the strict-white preset."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

CLAM_URL = "https://github.com/mahmoodlab/CLAM.git"
PRESET_NAME = "sln_strict_white.csv"


def setup_clam(clam_dir: Path, preset_src: Path, *, force: bool = False) -> Path:
    clam_dir = clam_dir.expanduser().resolve()
    clam_dir.parent.mkdir(parents=True, exist_ok=True)

    if clam_dir.exists() and force:
        print(f"[setup] removing existing {clam_dir}")
        shutil.rmtree(clam_dir)

    if not clam_dir.exists():
        print(f"[setup] cloning {CLAM_URL} → {clam_dir}")
        subprocess.run(
            ["git", "clone", "--depth", "1", CLAM_URL, str(clam_dir)],
            check=True,
        )
    else:
        print(f"[setup] CLAM already present: {clam_dir}")

    dest = clam_dir / "presets" / PRESET_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(preset_src, dest)
    print(f"[setup] preset installed: {dest}")
    return clam_dir


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clam-dir",
        type=Path,
        default=root.parents[0] / "mil-set" / "CLAM",
    )
    p.add_argument(
        "--preset",
        type=Path,
        default=root / "configs" / PRESET_NAME,
    )
    p.add_argument("--force", action="store_true", help="Re-clone CLAM.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.preset.is_file():
        raise FileNotFoundError(args.preset)
    setup_clam(args.clam_dir, args.preset, force=args.force)


if __name__ == "__main__":
    main()
