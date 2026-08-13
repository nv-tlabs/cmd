"""Convert downloaded DL3DV frames into Cosmos-Predict2.5 latent shards."""

import argparse
import fcntl
import json
import os
import re
import sys
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def environment_int(*names, default):
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def clip_key(scene_id, frame_indices):
    return f"{scene_id}/frame_{frame_indices[0]:05d}-frame_{frame_indices[-1]:05d}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create sharded LMDB training data from the DL3DV 480P "
            "images+poses release."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/dl3dv_1k_480p/1K"),
        help="Directory containing one subdirectory per DL3DV scene.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cosmos_i2v_lmdb"),
        help="Directory in which new or resumed LMDB shards are written.",
    )
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument(
        "--clip-stride",
        type=int,
        default=93,
        help="Input-frame distance between clip starts.",
    )
    parser.add_argument("--samples-per-shard", type=int, default=256)
    parser.add_argument("--map-size-gb", type=int, default=8)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=environment_int("WORLD_SIZE", "SLURM_NTASKS", default=1),
        help="Total preprocessing workers (inferred from torchrun or Slurm).",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=environment_int("RANK", "SLURM_PROCID", default=0),
        help="This worker's global ID (inferred from torchrun or Slurm).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Validate existing shards, skip their source clips, and continue "
            "the last partial shard."
        ),
    )
    parser.add_argument(
        "--captions-json",
        type=Path,
        required=True,
        help="JSON object mapping every processed scene hash to its prompt.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="Process only this many scenes (useful for a smoke test).",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        help="Stop after writing this many clips.",
    )
    parser.add_argument(
        "--device",
        help="Torch device; defaults to cuda:LOCAL_RANK or cuda:SLURM_LOCALID.",
    )
    parser.add_argument(
        "--model-name",
        default="nvidia/Cosmos-Predict2.5-2B",
    )
    return parser.parse_args()


def validate_args(args):
    import torch

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"DL3DV input directory not found: {args.input_dir}")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {args.output_dir}")
    for name in ("width", "height"):
        value = getattr(args, name)
        if value <= 0 or value % 8:
            raise ValueError(f"--{name} must be a positive multiple of 8")
    for name in ("num_frames", "clip_stride", "samples_per_shard", "map_size_gb"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_frames % 4 != 1:
        raise ValueError("--num-frames must equal 4k+1 for the Cosmos video VAE")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise ValueError("--max-scenes must be positive")
    if args.max_clips is not None and args.max_clips <= 0:
        raise ValueError("--max-clips must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError("--worker-id must be in [0, --num-workers)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but is not available")


def load_captions(path):
    with path.open(encoding="utf-8") as handle:
        captions = json.load(handle)
    if not isinstance(captions, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in captions.items()
    ):
        raise ValueError("--captions-json must contain a JSON object of string captions")
    return captions


def acquire_run_lock(output_dir):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.prepare.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(
            f"Another preprocessing process holds {lock_path}. "
            "Only one writer may use an output directory."
        ) from error
    return lock_handle


def scene_data(scene_dir):
    transforms_path = scene_dir / "transforms.json"
    image_dir = scene_dir / "images_8"
    if not transforms_path.is_file() or not image_dir.is_dir():
        return None

    with transforms_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    transforms = {
        Path(frame["file_path"]).name: np.asarray(
            frame["transform_matrix"], dtype=np.float32
        )
        for frame in metadata["frames"]
    }
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and path.name in transforms
    )
    if not image_paths:
        return None
    poses = np.stack([transforms[path.name] for path in image_paths])
    return metadata, image_paths, poses


def resize_crop(image, width, height):
    source_width, source_height = image.size
    scale = max(width / source_width, height / source_height)
    resized_width = max(width, round(source_width * scale))
    resized_height = max(height, round(source_height * scale))
    image = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return image.crop((left, top, left + width, top + height)), (
        resized_width,
        resized_height,
        left,
        top,
    )


def adjusted_intrinsics(metadata, geometry):
    resized_width, resized_height, left, top = geometry
    scale_x = resized_width / metadata["w"]
    scale_y = resized_height / metadata["h"]
    return np.asarray(
        [
            [metadata["fl_x"] * scale_x, 0.0, metadata["cx"] * scale_x - left],
            [0.0, metadata["fl_y"] * scale_y, metadata["cy"] * scale_y - top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_clip(image_paths, width, height):
    frames = []
    geometry = None
    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image, current_geometry = resize_crop(image, width, height)
            if geometry is None:
                geometry = current_geometry
            elif geometry != current_geometry:
                raise ValueError(f"Frame geometry changed within clip: {path}")
            frames.append(np.asarray(image, dtype=np.uint8))
    return np.stack(frames), geometry


def read_shape(txn, key, shard_path):
    value = txn.get(key.encode())
    if value is None:
        raise ValueError(f"Missing {key} in {shard_path}")
    try:
        return tuple(map(int, value.decode().split()))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Invalid {key} in {shard_path}: {value!r}") from error


def inspect_legacy_shards(
    output_dir,
    samples_per_shard,
    expected_latent_shape,
    num_frames,
):
    """Read and validate root-level shards produced by the single worker."""
    shard_pattern = re.compile(r"shard_(\d{5})")
    shards = []
    for path in output_dir.iterdir():
        match = shard_pattern.fullmatch(path.name)
        if match is not None:
            if not path.is_dir():
                raise ValueError(f"Legacy shard is not a directory: {path}")
            shards.append((int(match.group(1)), path))
    shards.sort()
    if not shards:
        return 0, None
    ids = [shard_id for shard_id, _ in shards]
    if ids != list(range(len(shards))):
        raise ValueError(f"Legacy shard IDs are not contiguous: {ids}")

    total = 0
    tail = None
    for position, (_, shard_path) in enumerate(shards):
        env = lmdb.open(
            str(shard_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin() as txn:
            latent_shape = read_shape(txn, "latents_shape", shard_path)
            count = latent_shape[0]
            expected_shapes = {
                "latents_shape": (count, *expected_latent_shape),
                "prompts_shape": (count,),
                "camera_poses_shape": (count, num_frames, 4, 4),
                "camera_intrinsics_shape": (count, 3, 3),
                "frame_indices_shape": (count, num_frames),
                "scene_ids_shape": (count,),
            }
            for key, expected in expected_shapes.items():
                actual = read_shape(txn, key, shard_path)
                if actual != expected:
                    raise ValueError(
                        f"Shape mismatch for {key} in {shard_path}: "
                        f"{actual}, expected {expected}"
                    )
            if count <= 0 or count > samples_per_shard:
                raise ValueError(f"Invalid sample count {count} in {shard_path}")
            if position != len(shards) - 1 and count != samples_per_shard:
                raise ValueError(
                    f"Non-final legacy shard {shard_path} has {count} samples; "
                    f"expected {samples_per_shard}"
                )
            if position == len(shards) - 1:
                index = count - 1
                scene = txn.get(f"scene_ids_{index}_data".encode())
                prompt = txn.get(f"prompts_{index}_data".encode())
                frames = txn.get(f"frame_indices_{index}_data".encode())
                if scene is None or prompt is None or frames is None:
                    raise ValueError(f"Incomplete final record in {shard_path}")
                tail = {
                    "scene_id": scene.decode(),
                    "prompt": prompt.decode(),
                    "frame_indices": np.frombuffer(frames, dtype=np.int32).copy(),
                }
        env.close()
        total += count
    return total, tail


def validate_worker_layout(output_dir, num_workers):
    pattern = re.compile(r"worker_(\d{5})_of_(\d{5})")
    for path in output_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        worker_id, recorded_workers = map(int, match.groups())
        if recorded_workers != num_workers:
            raise ValueError(
                f"Existing distributed output {path.name} used "
                f"{recorded_workers} workers, but this run uses {num_workers}. "
                "Resume with the original worker count."
            )
        if worker_id >= num_workers:
            raise ValueError(f"Invalid distributed worker directory: {path}")


class ShardWriter:
    def __init__(
        self,
        output_dir,
        samples_per_shard,
        map_size,
        expected_latent_shape,
        num_frames,
        resume=False,
    ):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.map_size = map_size
        self.expected_latent_shape = tuple(expected_latent_shape)
        self.num_frames = num_frames
        self.env = None
        self.shard_id = -1
        self.local_index = 0
        self.total = 0
        self.initial_total = 0
        self.resume_tail = None
        self._first_resumed_write = resume

        entries = sorted(self.output_dir.iterdir())
        if entries and not resume:
            raise FileExistsError(
                f"Output directory is not empty: {self.output_dir}. "
                "Pass --resume to validate and continue it, or choose a new directory."
            )
        if resume and entries:
            self._resume(entries)

    def _resume(self, entries):
        shard_pattern = re.compile(r"shard_(\d{5})")
        shards = []
        for entry in entries:
            match = shard_pattern.fullmatch(entry.name)
            if not entry.is_dir() or match is None:
                raise ValueError(
                    f"Unexpected entry in resumable LMDB directory: {entry}. "
                    "Only shard_XXXXX directories are allowed."
                )
            shards.append((int(match.group(1)), entry))
        shards.sort()
        expected_ids = list(range(len(shards)))
        shard_ids = [shard_id for shard_id, _ in shards]
        if shard_ids != expected_ids:
            raise ValueError(
                f"LMDB shards must be contiguous from shard_00000; found {shard_ids}"
            )

        counts = []
        last_map_size = self.map_size
        for position, (shard_id, shard_path) in enumerate(shards):
            env = lmdb.open(
                str(shard_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            last_map_size = max(last_map_size, env.info()["map_size"])
            with env.begin() as txn:
                raw_latent_shape = txn.get(b"latents_shape")
                if raw_latent_shape is None:
                    if position != len(shards) - 1:
                        raise ValueError(f"Empty shard is not last: {shard_path}")
                    count = 0
                else:
                    latent_shape = read_shape(txn, "latents_shape", shard_path)
                    expected = (latent_shape[0], *self.expected_latent_shape)
                    if latent_shape != expected:
                        raise ValueError(
                            f"Latent shape mismatch in {shard_path}: "
                            f"{latent_shape}, expected {expected}"
                        )
                    count = latent_shape[0]
                    expected_shapes = {
                        "prompts_shape": (count,),
                        "camera_poses_shape": (count, self.num_frames, 4, 4),
                        "camera_intrinsics_shape": (count, 3, 3),
                        "frame_indices_shape": (count, self.num_frames),
                        "scene_ids_shape": (count,),
                    }
                    for key, expected_shape in expected_shapes.items():
                        actual_shape = read_shape(txn, key, shard_path)
                        if actual_shape != expected_shape:
                            raise ValueError(
                                f"Shape mismatch for {key} in {shard_path}: "
                                f"{actual_shape}, expected {expected_shape}"
                            )

                    if count <= 0 or count > self.samples_per_shard:
                        raise ValueError(
                            f"Invalid sample count {count} in {shard_path}; expected "
                            f"1..{self.samples_per_shard}"
                        )
                    last_index = count - 1
                    tail_keys = [
                        "latents",
                        "prompts",
                        "camera_poses",
                        "camera_intrinsics",
                        "frame_indices",
                        "scene_ids",
                    ]
                    missing = [
                        key
                        for key in tail_keys
                        if txn.get(f"{key}_{last_index}_data".encode()) is None
                    ]
                    if missing:
                        raise ValueError(
                            f"Incomplete final record in {shard_path}: missing {missing}"
                        )
                    if position == len(shards) - 1:
                        self.resume_tail = {
                            "scene_id": txn.get(
                                f"scene_ids_{last_index}_data".encode()
                            ).decode(),
                            "prompt": txn.get(
                                f"prompts_{last_index}_data".encode()
                            ).decode(),
                            "frame_indices": np.frombuffer(
                                txn.get(f"frame_indices_{last_index}_data".encode()),
                                dtype=np.int32,
                            ).copy(),
                        }
            env.close()
            counts.append(count)

            if position != len(shards) - 1 and count != self.samples_per_shard:
                raise ValueError(
                    f"Non-final shard {shard_path} has {count} samples; "
                    f"expected {self.samples_per_shard}"
                )

        self.total = sum(counts)
        self.initial_total = self.total
        self.shard_id = shards[-1][0]
        self.local_index = counts[-1]
        self.map_size = last_map_size
        if self.local_index < self.samples_per_shard:
            self.env = self._open_env(shards[-1][1])
        print(
            f"Resume validated {len(shards)} shard(s) with {self.total} "
            f"existing clips; next output index is {self.total}."
        )

    def _open_env(self, shard_path):
        return lmdb.open(
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

    def _open_next(self):
        if self.env is not None:
            self.env.sync()
            self.env.close()
        self.shard_id += 1
        self.local_index = 0
        shard_path = self.output_dir / f"shard_{self.shard_id:05d}"
        self.env = self._open_env(shard_path)

    def write(self, latent, prompt, poses, intrinsics, frame_indices, scene_id):
        if self.env is None or self.local_index == self.samples_per_shard:
            self._open_next()
        index = self.local_index
        with self.env.begin(write=True) as txn:
            if self._first_resumed_write:
                current_shape = txn.get(b"latents_shape")
                current_count = (
                    tuple(map(int, current_shape.decode().split()))[0]
                    if current_shape is not None
                    else 0
                )
                if current_count != index:
                    raise RuntimeError(
                        "The last shard changed after resume validation. "
                        "Another preprocessing process may still be writing it; "
                        "stop that process before resuming."
                    )
                self._first_resumed_write = False
            values = {
                f"latents_{index}_data": latent.tobytes(),
                f"prompts_{index}_data": prompt.encode("utf-8"),
                f"camera_poses_{index}_data": poses.tobytes(),
                f"camera_intrinsics_{index}_data": intrinsics.tobytes(),
                f"frame_indices_{index}_data": frame_indices.tobytes(),
                f"scene_ids_{index}_data": scene_id.encode("utf-8"),
            }
            for key, value in values.items():
                txn.put(key.encode(), value)

            count = index + 1
            shapes = {
                "latents_shape": (count, *latent.shape),
                "prompts_shape": (count,),
                "camera_poses_shape": (count, *poses.shape),
                "camera_intrinsics_shape": (count, *intrinsics.shape),
                "frame_indices_shape": (count, *frame_indices.shape),
                "scene_ids_shape": (count,),
            }
            for key, shape in shapes.items():
                txn.put(key.encode(), " ".join(map(str, shape)).encode())

        self.local_index += 1
        self.total += 1

    def close(self):
        if self.env is not None:
            self.env.sync()
            self.env.close()
            self.env = None


def main():
    args = parse_args()
    import torch

    if args.device is None:
        local_rank = environment_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
        args.device = f"cuda:{local_rank}"
    validate_args(args)
    captions = load_captions(args.captions_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if args.max_scenes is not None:
        scene_dirs = scene_dirs[: args.max_scenes]

    expected_shape = (
        (args.num_frames - 1) // 4 + 1,
        16,
        args.height // 8,
        args.width // 8,
    )
    validate_worker_layout(args.output_dir, args.num_workers)
    if args.num_workers == 1:
        worker_output_dir = args.output_dir
        legacy_total = 0
        legacy_tail = None
    else:
        legacy_total, legacy_tail = inspect_legacy_shards(
            output_dir=args.output_dir,
            samples_per_shard=args.samples_per_shard,
            expected_latent_shape=expected_shape,
            num_frames=args.num_frames,
        )
        worker_output_dir = args.output_dir / (
            f"worker_{args.worker_id:05d}_of_{args.num_workers:05d}"
        )
        worker_output_dir.mkdir(parents=True, exist_ok=True)

    if not args.resume and legacy_total > 0:
        raise FileExistsError(
            f"Existing single-worker output found in {args.output_dir}. "
            "Pass --resume to continue it."
        )

    run_lock = acquire_run_lock(worker_output_dir)
    writer = ShardWriter(
        output_dir=worker_output_dir,
        samples_per_shard=args.samples_per_shard,
        map_size=args.map_size_gb * 1024**3,
        expected_latent_shape=expected_shape,
        num_frames=args.num_frames,
        resume=args.resume,
    )
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = None
    resume_tail_validated = writer.initial_total == 0
    legacy_tail_validated = legacy_total == 0
    source_clip_index = 0
    worker_clip_index = 0

    try:
        for scene_dir in tqdm(
            scene_dirs,
            desc=f"DL3DV worker {args.worker_id}/{args.num_workers}",
        ):
            loaded = scene_data(scene_dir)
            if loaded is None:
                tqdm.write(f"Skipping incomplete scene: {scene_dir.name}")
                continue
            metadata, image_paths, poses = loaded

            starts = range(
                0,
                len(image_paths) - args.num_frames + 1,
                args.clip_stride,
            )
            for start in starts:
                stop = start + args.num_frames
                clip_paths = image_paths[start:stop]
                frame_indices = np.asarray(
                    [int(path.stem.rsplit("_", 1)[-1]) for path in clip_paths],
                    dtype=np.int32,
                )
                key = clip_key(scene_dir.name, frame_indices)
                if key not in captions:
                    raise KeyError(
                        f"No Qwen caption for DL3DV clip {key} in "
                        f"{args.captions_json}. Generic prompt fallback is disabled."
                    )
                prompt = captions[key]
                source_clip_index += 1

                if args.max_clips is not None and source_clip_index > args.max_clips:
                    break

                if source_clip_index <= legacy_total:
                    if source_clip_index == legacy_total:
                        if legacy_tail is None:
                            raise ValueError("Legacy resume tail metadata is missing")
                        if (
                            legacy_tail["scene_id"] != scene_dir.name
                            or legacy_tail["prompt"] != prompt
                            or not np.array_equal(
                                legacy_tail["frame_indices"], frame_indices
                            )
                        ):
                            raise ValueError(
                                "The existing single-worker prefix does not match "
                                "the current preprocessing inputs or arguments at "
                                f"source clip {source_clip_index}."
                            )
                        legacy_tail_validated = True
                    continue

                distributed_index = source_clip_index - legacy_total - 1
                if distributed_index % args.num_workers != args.worker_id:
                    continue
                worker_clip_index += 1

                if worker_clip_index <= writer.initial_total:
                    if worker_clip_index == writer.initial_total:
                        tail = writer.resume_tail
                        if tail is None:
                            raise ValueError("Resume tail metadata is missing")
                        if (
                            tail["scene_id"] != scene_dir.name
                            or tail["prompt"] != prompt
                            or not np.array_equal(tail["frame_indices"], frame_indices)
                        ):
                            raise ValueError(
                                "The existing output does not match the current "
                                "input order, clip settings, or captions at clip "
                                f"{source_clip_index}. Use the same worker count and "
                                "preprocessing "
                                "arguments that created the existing shards."
                            )
                        resume_tail_validated = True
                    continue

                clip_frames, geometry = load_clip(
                    clip_paths, args.width, args.height
                )
                intrinsics = adjusted_intrinsics(metadata, geometry)
                if vae is None:
                    from cosmos import CosmosVAEWrapper

                    vae = CosmosVAEWrapper(model_name=args.model_name).to(
                        device=device, dtype=dtype
                    ).eval()
                pixels = (
                    torch.from_numpy(clip_frames)
                    .permute(3, 0, 1, 2)
                    .unsqueeze(0)
                    .to(device=device, dtype=dtype)
                    .div_(127.5)
                    .sub_(1.0)
                )
                with torch.inference_mode():
                    latent = vae.encode_to_latent(pixels)[0]
                latent = latent.to(device="cpu", dtype=torch.float16).numpy()
                if latent.shape != expected_shape:
                    raise ValueError(
                        f"Unexpected latent shape {latent.shape}; expected "
                        f"{expected_shape} for scene {scene_dir.name}"
                    )
                if not np.isfinite(latent).all():
                    raise ValueError(f"Non-finite latent in scene {scene_dir.name}")

                writer.write(
                    latent=latent,
                    prompt=prompt,
                    poses=poses[start:stop],
                    intrinsics=intrinsics,
                    frame_indices=frame_indices,
                    scene_id=scene_dir.name,
                )
                del pixels, latent

            if args.max_clips is not None and source_clip_index >= args.max_clips:
                break
    finally:
        writer.close()
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_UN)
        run_lock.close()

    if not resume_tail_validated:
        raise RuntimeError(
            f"Existing output contains {writer.initial_total} clips, but the "
            "selected input contains fewer clips. Check --input-dir, "
            "--max-scenes, --num-frames, and --clip-stride."
        )
    if not legacy_tail_validated:
        raise RuntimeError(
            f"The existing single-worker prefix contains {legacy_total} clips, "
            "but the selected input contains fewer clips."
        )
    if writer.total == 0 and legacy_total == 0:
        raise RuntimeError(f"Worker {args.worker_id} found no complete clips")
    print(
        f"Worker {args.worker_id}/{args.num_workers} contains {writer.total} "
        f"distributed clips in {worker_output_dir}; retained {legacy_total} "
        f"single-worker prefix clips and added "
        f"{writer.total - writer.initial_total} clips in this run."
    )


if __name__ == "__main__":
    main()
