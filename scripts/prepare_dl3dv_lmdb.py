#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Encode one local DL3DV MP4 per scene into Cosmos LMDB shards."""

from __future__ import annotations

import argparse
import json
import lmdb
import os
import shutil
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cosmos import CosmosVAEWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/dl3dv_1k_videos/1K"),
        help=(
            "Official DL3DV video subset directory containing "
            "<scene-id>/video.mp4."
        ),
    )
    parser.add_argument(
        "--camera-dir",
        type=Path,
        default=Path("data/dl3dv_1k_vipe_dav3_cameras"),
        help="Local directory containing pose/ and intrinsics/ NPZ files.",
    )
    parser.add_argument(
        "--camera-conditioning",
        action="store_true",
        help="Write matching pose and intrinsic arrays into the LMDB.",
    )
    parser.add_argument(
        "--captions-json",
        type=Path,
        default=Path("data/dl3dv_captions.json"),
    )
    parser.add_argument(
        "--latent-frames",
        type=int,
        choices=(24, 32),
        default=32,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to tN_camera with cameras, otherwise tN.",
    )
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sample-width", type=int, default=854)
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Calibration width used by the supplied intrinsics.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Calibration height used by the supplied intrinsics.",
    )
    parser.add_argument("--samples-per-shard", type=int, default=256)
    parser.add_argument("--map-size-gb", type=int, default=8)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start at this index in the sorted camera-matched scene list.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="Process at most this many scenes; useful for a smoke test.",
    )
    parser.add_argument(
        "--model-name",
        default="nvidia/Cosmos-Predict2.5-2B",
    )
    parser.add_argument(
        "--ffmpeg-path",
        type=Path,
        default=Path(shutil.which("ffmpeg") or "/usr/bin/ffmpeg"),
        help="CUDA-enabled FFmpeg executable used to decode and resize frames.",
    )
    return parser.parse_args()


def complete_camera_keys(camera_dir: Path) -> list[str]:
    pose_keys = {path.stem for path in (camera_dir / "pose").glob("*.npz")}
    intrinsic_keys = {
        path.stem for path in (camera_dir / "intrinsics").glob("*.npz")
    }
    keys = sorted(pose_keys & intrinsic_keys)
    if len(keys) != 974:
        raise ValueError(f"Expected 974 complete camera pairs, found {len(keys)}")
    return keys


def prompts_by_scene(path: Path) -> dict[str, str]:
    values = json.loads(path.read_text())
    prompts: dict[str, str] = {}
    for clip_key, prompt in sorted(values.items()):
        scene_id, _, _ = clip_key.partition("/")
        if scene_id not in prompts:
            prompts[scene_id] = prompt
    return prompts


def load_camera(path: Path, expected_key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as values:
        data = values["data"].astype(np.float32, copy=True)
        inds = values["inds"].astype(np.int64, copy=True)
    expected = np.arange(len(inds), dtype=np.int64)
    if not np.array_equal(inds, expected):
        raise ValueError(f"Non-contiguous camera indices in {path} for {expected_key}")
    return data, inds


def load_camera_frame_count(path: Path, expected_key: str) -> int:
    with np.load(path) as values:
        inds = values["inds"].astype(np.int64, copy=False)
        expected = np.arange(len(inds), dtype=np.int64)
        if not np.array_equal(inds, expected):
            raise ValueError(
                f"Non-contiguous camera indices in {path} for {expected_key}"
            )
        return len(inds)


def camera_aligned_source_indices(
    video_frames: int,
    camera_frames: int,
    requested_frames: int,
) -> np.ndarray:
    """Map camera-timeline positions to their source MP4 frame indices."""
    if not 0 < requested_frames <= camera_frames <= video_frames:
        raise ValueError(
            "Invalid video/camera frame counts: "
            f"video={video_frames}, camera={camera_frames}, "
            f"requested={requested_frames}"
        )

    # Preserve an exact integer decimation when the camera extraction used
    # one (for example, 60 fps video -> 20 fps cameras). Otherwise, map by
    # relative timeline position to support arbitrary source/camera FPS pairs.
    stride = max(1, round(video_frames / camera_frames))
    positions = np.arange(requested_frames, dtype=np.int64)
    if (video_frames + stride - 1) // stride == camera_frames:
        source_indices = positions * stride
    else:
        source_indices = positions * video_frames // camera_frames

    if source_indices[-1] >= video_frames or np.any(np.diff(source_indices) <= 0):
        raise ValueError(
            "Could not construct a strictly increasing camera-aligned frame "
            f"map for video={video_frames}, camera={camera_frames}"
        )
    return source_indices


def decode_video_clip(
    path: Path,
    clip_frames: int,
    camera_frames: int | None,
    sample_h: int,
    sample_w: int,
    crop_h: int,
    crop_w: int,
    ffmpeg_path: Path,
    gpu_index: int,
) -> tuple[torch.Tensor, int, int, int, float, int, int]:
    """Decode selected frames on CUDA and pipe only resized RGB frames to Python."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total = int(stream.frames or 0)
        if total <= 0:
            raise ValueError(f"MP4 does not report its full frame count: {path}")
        source_h = int(stream.codec_context.height)
        source_w = int(stream.codec_context.width)
        if source_h <= 0 or source_w <= 0:
            raise ValueError(f"MP4 does not report valid dimensions: {path}")
        source_indices = (
            np.arange(clip_frames, dtype=np.int64)
            if camera_frames is None
            else camera_aligned_source_indices(
                total,
                camera_frames,
                clip_frames,
            )
        )

    scale = max(sample_h / source_h, sample_w / source_w)
    resized_h = int(round(source_h * scale))
    resized_w = int(round(source_w * scale))
    top = (resized_h - crop_h) // 2
    left = (resized_w - crop_w) // 2
    if top < 0 or left < 0:
        raise ValueError(
            f"Resize {resized_w}x{resized_h} is smaller than "
            f"crop {crop_w}x{crop_h} for {path}"
        )
    if resized_h % 2 or resized_w % 2:
        raise ValueError(
            f"CUDA yuv420p resize requires even dimensions, got "
            f"{resized_w}x{resized_h} for {path}"
        )

    # The commas in eq(n,index) must be escaped for libavfilter. Selecting
    # before scale_cuda means skipped source frames are decoded on NVDEC but
    # never resized or copied back to host memory.
    select_expr = "+".join(f"eq(n\\,{int(index)})" for index in source_indices)
    video_filter = (
        f"select={select_expr},"
        f"scale_cuda={resized_w}:{resized_h}:format=yuv420p,"
        "hwdownload,format=yuv420p,format=rgb24"
    )
    command = [
        str(ffmpeg_path),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-hwaccel_device",
        str(gpu_index),
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-an",
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(clip_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    frame_bytes = resized_h * resized_w * 3
    frames = torch.empty(
        (clip_frames, 3, crop_h, crop_w),
        dtype=torch.uint8,
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_bytes * 2,
    )
    assert process.stdout is not None and process.stderr is not None
    try:
        for output_index in range(clip_frames):
            payload = bytearray(frame_bytes)
            view = memoryview(payload)
            received = 0
            while received < frame_bytes:
                count = process.stdout.readinto(view[received:])
                if not count:
                    break
                received += count
            if received != frame_bytes:
                _, stderr = process.communicate()
                details = stderr.decode(errors="replace").strip()
                raise RuntimeError(
                    f"CUDA FFmpeg yielded {output_index} of {clip_frames} "
                    f"frames for {path} (exit {process.returncode}):\n{details}"
                )
            array = np.frombuffer(payload, dtype=np.uint8).reshape(
                resized_h,
                resized_w,
                3,
            )
            cropped = array[top : top + crop_h, left : left + crop_w]
            frames[output_index].copy_(torch.from_numpy(cropped).permute(2, 0, 1))

        stdout_tail, stderr = process.communicate()
        details = stderr.decode(errors="replace").strip()
        if process.returncode != 0:
            raise RuntimeError(
                f"CUDA FFmpeg failed for {path} (exit {process.returncode}):\n"
                f"{details}"
            )
        if stdout_tail:
            raise RuntimeError(f"CUDA FFmpeg yielded extra frame bytes for {path}")
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.communicate()
        raise

    return frames, total, source_h, source_w, scale, left, top


def camera_matrices(
    values: np.ndarray,
    source_h: int,
    source_w: int,
    camera_h: int,
    camera_w: int,
    scale: float,
    left: int,
    top: int,
) -> np.ndarray:
    scale_x = source_w / camera_w * scale
    scale_y = source_h / camera_h * scale
    if values.ndim == 3 and values.shape[-2:] == (3, 3):
        matrices = values.astype(np.float32, copy=True)
        matrices[:, 0, 0] *= scale_x
        matrices[:, 1, 1] *= scale_y
        matrices[:, 0, 2] = matrices[:, 0, 2] * scale_x - left
        matrices[:, 1, 2] = matrices[:, 1, 2] * scale_y - top
        return matrices
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError(f"Unsupported intrinsic shape: {values.shape}")
    matrices = np.zeros((len(values), 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = values[:, 0] * scale_x
    matrices[:, 1, 1] = values[:, 1] * scale_y
    matrices[:, 0, 2] = values[:, 2] * scale_x - left
    matrices[:, 1, 2] = values[:, 3] * scale_y - top
    matrices[:, 2, 2] = 1.0
    return matrices


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        samples_per_shard: int,
        map_size: int,
        expected_latent_shape: tuple[int, ...],
    ) -> None:
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.map_size = map_size
        self.expected_latent_shape = expected_latent_shape
        self.env = None
        self.shard_id = -1
        self.local_index = 0
        self.total = 0

    def _open_next(self) -> None:
        if self.env is not None:
            self.env.sync()
            self.env.close()
        self.shard_id += 1
        self.local_index = 0
        shard_path = self.output_dir / f"shard_{self.shard_id:05d}"
        self.env = lmdb.open(
            str(shard_path),
            map_size=self.map_size,
            subdir=True,
            readonly=False,
            metasync=True,
            sync=True,
            lock=True,
            readahead=False,
            meminit=False,
        )

    def write(
        self,
        latent: np.ndarray,
        prompt: str,
        scene_id: str,
        poses: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> None:
        if latent.shape != self.expected_latent_shape:
            raise ValueError(
                f"Unexpected latent shape {latent.shape}; "
                f"expected {self.expected_latent_shape}"
            )
        if (poses is None) != (intrinsics is None):
            raise ValueError("Poses and intrinsics must be supplied together")
        if self.env is None or self.local_index == self.samples_per_shard:
            self._open_next()

        index = self.local_index
        values = {
            "latents": latent,
            "prompts": prompt,
            "scene_ids": scene_id,
        }
        if poses is not None:
            values["camera_poses"] = poses
            values["camera_intrinsics"] = intrinsics

        with self.env.begin(write=True) as txn:
            for name, value in values.items():
                payload = value.encode() if isinstance(value, str) else value.tobytes()
                txn.put(f"{name}_{index}_data".encode(), payload)

            count = index + 1
            shapes = {
                "latents_shape": (count, *latent.shape),
                "prompts_shape": (count,),
                "scene_ids_shape": (count,),
            }
            if poses is not None:
                shapes["camera_poses_shape"] = (count, *poses.shape)
                shapes["camera_intrinsics_shape"] = (count, *intrinsics.shape)
            for name, shape in shapes.items():
                txn.put(name.encode(), " ".join(map(str, shape)).encode())

        self.local_index += 1
        self.total += 1

    def close(self) -> None:
        if self.env is not None:
            self.env.sync()
            self.env.close()
            self.env = None


def main() -> None:
    args = parse_args()
    clip_frames = 4 * args.latent_frames - 3
    if args.output_dir is None:
        suffix = "_camera" if args.camera_conditioning else ""
        args.output_dir = Path(f"data/cosmos_i2v_lmdb_t{args.latent_frames}{suffix}")
    if args.camera_conditioning and args.latent_frames != 32:
        raise ValueError("The camera-conditioned dataset is t32")
    if not args.camera_conditioning and args.latent_frames != 24:
        raise ValueError("The non-camera dataset is t24")
    if (args.height, args.width, args.sample_width) != (480, 832, 854):
        raise ValueError("The Cosmos configs require resize-to-480x854 then 480x832 crop")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VAE encoding")
    if not args.ffmpeg_path.is_file():
        raise FileNotFoundError(f"FFmpeg executable not found: {args.ffmpeg_path}")

    video_keys = sorted(
        path.name
        for path in args.video_dir.iterdir()
        if path.is_dir() and (path / "video.mp4").is_file()
    )
    camera_keys = complete_camera_keys(args.camera_dir)
    missing = sorted(set(camera_keys) - set(video_keys))
    if missing:
        raise ValueError(
            f"Missing {len(missing)} camera-matched videos; "
            f"first missing scene: {missing[0]}"
        )
    keys = camera_keys
    if not 0 <= args.start_index < len(keys):
        raise ValueError(
            f"--start-index must be between 0 and {len(keys) - 1}"
        )
    keys = keys[args.start_index :]
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("--max-scenes must be at least 1")
        keys = keys[: args.max_scenes]
    prompts = prompts_by_scene(args.captions_json)
    missing_prompts = [key for key in keys if key not in prompts]
    if missing_prompts:
        raise KeyError(f"Missing prompts for {len(missing_prompts)} scenes")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    worker_keys = keys[rank::world_size]
    worker_dir = args.output_dir / f"worker_{rank:05d}_of_{world_size:05d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    if any(worker_dir.iterdir()):
        raise FileExistsError(f"Fresh output required: {worker_dir} is not empty")

    expected_latent_shape = (
        args.latent_frames,
        16,
        args.height // 8,
        args.width // 8,
    )
    writer = ShardWriter(
        output_dir=worker_dir,
        samples_per_shard=args.samples_per_shard,
        map_size=args.map_size_gb * 1024**3,
        expected_latent_shape=expected_latent_shape,
    )
    device = torch.device(f"cuda:{local_rank}")
    vae = CosmosVAEWrapper(model_name=args.model_name).to(
        device=device,
        dtype=torch.bfloat16,
    ).eval()

    try:
        for key in tqdm(
            worker_keys,
            desc=f"DL3DV t{args.latent_frames} worker {rank}/{world_size}",
            position=rank,
        ):
            poses = None
            intrinsics = None
            if args.camera_conditioning:
                poses, pose_inds = load_camera(
                    args.camera_dir / "pose" / f"{key}.npz",
                    key,
                )
                intrinsics, intrinsic_inds = load_camera(
                    args.camera_dir / "intrinsics" / f"{key}.npz",
                    key,
                )
                if len(intrinsic_inds) != len(pose_inds):
                    raise ValueError(
                        f"Pose/intrinsic length mismatch for {key}: "
                        f"poses={len(pose_inds)}, intrinsics={len(intrinsic_inds)}"
                    )
                camera_frames = len(pose_inds)
            else:
                camera_frames = load_camera_frame_count(
                    args.camera_dir / "pose" / f"{key}.npz",
                    key,
                )

            video, _, source_h, source_w, scale, left, top = decode_video_clip(
                args.video_dir / key / "video.mp4",
                clip_frames,
                camera_frames,
                args.height,
                args.sample_width,
                args.height,
                args.width,
                args.ffmpeg_path,
                local_rank,
            )
            clip_poses = None
            clip_intrinsics = None
            if args.camera_conditioning:
                clip_poses = poses[:clip_frames]
                clip_intrinsics = camera_matrices(
                    intrinsics[:clip_frames],
                    source_h,
                    source_w,
                    args.camera_height,
                    args.camera_width,
                    scale,
                    left,
                    top,
                )
            pixels = (
                video.permute(1, 0, 2, 3)
                .unsqueeze(0)
                .to(device=device, dtype=torch.bfloat16)
                .div_(127.5)
                .sub_(1.0)
            )
            with torch.inference_mode():
                latent = vae.encode_to_latent(pixels)[0]
            latent = latent.to(device="cpu", dtype=torch.float16).numpy()
            if latent.shape != expected_latent_shape:
                raise ValueError(
                    f"Unexpected latent shape for {key}: {latent.shape}; "
                    f"expected {expected_latent_shape}"
                )
            writer.write(
                latent=latent,
                prompt=prompts[key],
                scene_id=key,
                poses=clip_poses,
                intrinsics=clip_intrinsics,
            )
            del video, pixels, latent
    finally:
        writer.close()

    print(
        f"Worker {rank}/{world_size} wrote {writer.total} local DL3DV "
        f"t{args.latent_frames} clips "
        f"({'camera' if args.camera_conditioning else 'no camera'})",
        flush=True,
    )


if __name__ == "__main__":
    main()
