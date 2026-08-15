# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Caption DL3DV training clips with Qwen3-VL-8B-Instruct."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import av
import numpy as np
from tqdm import tqdm


CAPTION_PROMPT = """The images are chronological samples from one video clip.
Write a 50-to-70-word caption for image-to-video model training using exactly
three complete sentences. In the first sentence describe the environment,
important visible objects, lighting, and spatial layout. In the second sentence
describe the camera movement and how the view changes. In the third sentence
add any remaining visually supported details. Do not mention images, frames,
sampling, or this request, and do not invent unsupported actions. Return only
the caption and end it with a period."""

MIN_CAPTION_WORDS = 25
MAX_CAPTION_WORDS = 90


def environment_int(*names, default):
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def clip_key(scene_id, frame_indices):
    return f"{scene_id}/frame_{frame_indices[0]:05d}-frame_{frame_indices[-1]:05d}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/dl3dv_1k_videos/1K"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dl3dv_captions.json"),
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--caption-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=environment_int("WORLD_SIZE", "SLURM_NTASKS", default=1),
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=environment_int("RANK", "SLURM_PROCID", default=0),
    )
    parser.add_argument("--device")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check that every selected clip has a caption without loading Qwen.",
    )
    return parser.parse_args()


def caption_is_valid(caption):
    word_count = len(caption.split())
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", caption))
    return (
        caption.endswith((".", "!", "?"))
        and MIN_CAPTION_WORDS <= word_count <= MAX_CAPTION_WORDS
        and 1 <= sentence_count <= 3
    )


def finalize_caption(caption):
    caption = " ".join(caption.strip().split())
    sentence_ends = [match.end() for match in re.finditer(r"[.!?](?:\s|$)", caption)]
    if not sentence_ends:
        raise RuntimeError("Qwen returned no complete sentence")
    if not caption.endswith((".", "!", "?")):
        caption = caption[: sentence_ends[-1]].strip()
        sentence_ends = [
            match.end() for match in re.finditer(r"[.!?](?:\s|$)", caption)
        ]
    if len(sentence_ends) > 3:
        caption = caption[: sentence_ends[2]].strip()
        sentence_ends = sentence_ends[:3]
    if len(caption.split()) > MAX_CAPTION_WORDS:
        allowed_ends = [
            end
            for end in sentence_ends
            if len(caption[:end].split()) <= MAX_CAPTION_WORDS
        ]
        if allowed_ends:
            caption = caption[: allowed_ends[-1]].strip()
    if not caption_is_valid(caption):
        raise RuntimeError(
            "Qwen caption failed quality validation: "
            f"{len(caption.split())} words, {caption!r}"
        )
    return caption


def load_captions(path):
    if not path.exists():
        return {}
    captions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(captions, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and caption_is_valid(value)
        for key, value in captions.items()
    ):
        raise ValueError(f"Invalid captions file: {path}")
    return captions


def all_clips(input_dir, num_frames, max_clips=None):
    clips = []
    for scene_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        video_path = scene_dir / "video.mp4"
        if not video_path.is_file():
            continue
        indices = np.arange(1, num_frames + 1, dtype=np.int32)
        clips.append(
            (clip_key(scene_dir.name, indices), video_path)
        )
        if max_clips is not None and len(clips) >= max_clips:
            return clips
    return clips


def sampled_images(video_path, count, num_frames):
    positions = set(
        np.linspace(0, num_frames - 1, min(count, num_frames), dtype=int)
    )
    images = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index in positions:
                images.append(frame.to_image().convert("RGB"))
            if index == num_frames - 1:
                break
    if len(images) != len(positions):
        raise ValueError(
            f"{video_path} has fewer than {num_frames} decoded frames"
        )
    return images


def caption_clip(
    model,
    processor,
    device,
    video_path,
    count,
    num_frames,
    max_new_tokens,
):
    images = sampled_images(video_path, count, num_frames)
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    generated = generated[:, inputs["input_ids"].shape[1] :]
    caption = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return finalize_caption(caption)


def store_caption(path, lock_path, key, caption):
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        captions = load_captions(path)
        captions[key] = caption
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(captions, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def cache_model(model_name, worker_id, num_workers):
    from huggingface_hub import snapshot_download

    lock_name = hashlib.sha256(model_name.encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"qwen_caption_{lock_name}.lock"
    print(
        f"[caption worker {worker_id}/{num_workers}] waiting for model cache",
        flush=True,
    )
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        model_path = snapshot_download(model_name)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    print(
        f"[caption worker {worker_id}/{num_workers}] checkpoint cached; "
        "loading model onto GPU",
        flush=True,
    )
    return model_path


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    if args.num_workers <= 0 or not 0 <= args.worker_id < args.num_workers:
        raise ValueError("Invalid worker ID/count")
    if args.caption_frames <= 0 or args.num_frames <= 0:
        raise ValueError("Frame counts must be positive")

    print(
        f"[caption worker {args.worker_id}/{args.num_workers}] scanning clips",
        flush=True,
    )
    clips = all_clips(
        args.input_dir,
        args.num_frames,
        args.max_clips,
    )
    captions = load_captions(args.output)
    missing = [key for key, _ in clips if key not in captions]
    if args.validate_only:
        if missing:
            raise RuntimeError(
                f"Missing {len(missing)} of {len(clips)} clip captions; "
                f"first missing key: {missing[0]}"
            )
        print(f"Validated {len(clips)} Qwen captions in {args.output}")
        return

    owned = [
        (index, key, video_path)
        for index, (key, video_path) in enumerate(clips)
        if index % args.num_workers == args.worker_id and key not in captions
    ]
    if not owned:
        print(f"Caption worker {args.worker_id}/{args.num_workers}: nothing to do")
        return
    print(
        f"[caption worker {args.worker_id}/{args.num_workers}] "
        f"found {len(clips)} clips; {len(owned)} assigned and unfinished",
        flush=True,
    )

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if args.device is None:
        local_rank = environment_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
        args.device = f"cuda:{local_rank}"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    model_path = cache_model(args.model_name, args.worker_id, args.num_workers)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    print(
        f"[caption worker {args.worker_id}/{args.num_workers}] model ready on "
        f"{device}",
        flush=True,
    )
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")

    for _, key, video_path in tqdm(
        owned,
        desc=f"Qwen captions {args.worker_id}/{args.num_workers}",
    ):
        with torch.inference_mode():
            caption = caption_clip(
                model,
                processor,
                device,
                video_path,
                args.caption_frames,
                args.num_frames,
                args.max_new_tokens,
            )
        store_caption(args.output, lock_path, key, caption)
    print(
        f"Caption worker {args.worker_id}/{args.num_workers} added "
        f"{len(owned)} captions to {args.output}"
    )


if __name__ == "__main__":
    main()
