#!/usr/bin/env python3
"""Step 1: download SLN-Breast from Hugging Face (no duplicate of existing files)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Refrainkana33/sln-breast-tcia-svs"


def download_sln_breast(
    output_dir: Path,
    *,
    labels_only: bool = False,
    repo_id: str = REPO_ID,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = ["target.csv"] if labels_only else None
    token = token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )

    print(f"[download] repo={repo_id}")
    print(f"[download] out={output_dir}")
    print(f"[download] mode={'labels_only' if labels_only else 'full (~56GB SVS)'}")

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        revision=revision,
        allow_patterns=allow_patterns,
        token=token,
        max_workers=8,
    )
    local = Path(local_path)
    n_svs = len(list(local.glob("*.svs")))
    labels = local / "target.csv"
    print(f"[download] done: {n_svs} svs; labels={labels.is_file()}")
    return local


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "sln-breast",
    )
    p.add_argument("--labels-only", action="store_true")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--revision", default=None)
    p.add_argument("--token", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    download_sln_breast(
        args.output_dir,
        labels_only=args.labels_only,
        repo_id=args.repo_id,
        revision=args.revision,
        token=args.token,
    )


if __name__ == "__main__":
    main()
