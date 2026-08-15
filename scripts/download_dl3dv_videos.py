#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Download the complete original DL3DV 1K video split."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dl3dv_1k_videos"),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_id = "DL3DV/DL3DV-ALL-video"
    repo_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    keys = sorted(
        path.split("/")[1]
        for path in repo_files
        if path.startswith("1K/") and path.endswith("/video.mp4")
    )
    if len(keys) != 1000 or len(set(keys)) != 1000:
        raise ValueError(f"Expected 1000 videos in the DL3DV 1K split, found {len(keys)}")

    def download_video(key: str) -> None:
        video_path = args.output_dir / "1K" / key / "video.mp4"
        if video_path.is_file():
            return
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"1K/{key}/video.mp4",
            local_dir=args.output_dir,
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(download_video, key): key for key in keys}
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 25 == 0 or completed == len(keys):
                print(f"Available {completed}/{len(keys)} videos", flush=True)

    print(f"All {len(keys)} DL3DV 1K videos are available in {args.output_dir}")


if __name__ == "__main__":
    main()
