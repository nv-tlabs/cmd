# SPDX-FileCopyrightText: Copyright (c) 2025 The Self-Forcing Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed


_COSMOS_CHECKPOINT_OPTIONAL_BUFFERS = {
    "model.accum_video_sample_counter",
    "model.accum_image_sample_counter",
    "model.accum_iteration",
    "model.accum_train_in_hours",
}


def load_generator_checkpoint(pipeline, checkpoint_path, model_family, use_ema):
    """Load either a training checkpoint or an exported flat safetensors DiT."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        if use_ema:
            raise ValueError("Exported safetensors contain one generator; do not use --use_ema")
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path), device="cpu")
    else:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        component = "generator_ema" if use_ema else "generator"
        if component in state_dict:
            state_dict = state_dict[component]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        elif use_ema:
            raise KeyError(f"{checkpoint_path} has no {component!r} component")

    if model_family.lower() != "cosmos":
        pipeline.generator.load_state_dict(state_dict, strict=True)
        return

    # Released Cosmos checkpoints contain bare DiT keys such as `blocks.0...`,
    # while the inference pipeline owns that DiT under `generator.model`.
    normalized = {}
    for name, value in state_dict.items():
        name = name.removeprefix("generator.")
        normalized_name = name if name.startswith("model.") else f"model.{name}"
        if normalized_name in normalized:
            raise RuntimeError(
                f"Duplicate checkpoint key after normalization: {normalized_name}"
            )
        normalized[normalized_name] = value

    incompatible = pipeline.generator.load_state_dict(normalized, strict=False)
    invalid_missing = set(incompatible.missing_keys) - _COSMOS_CHECKPOINT_OPTIONAL_BUFFERS
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible Cosmos generator checkpoint: "
            f"missing={sorted(invalid_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    print(f"Loaded {checkpoint_path}: {len(normalized)} generator tensors")


def load_camera_inputs(camera_path, latent_frames, frame_stride, batch_size):
    with np.load(camera_path, allow_pickle=False) as camera:
        world_to_camera = np.asarray(camera["target_w2c"], dtype=np.float32)
        intrinsics = np.asarray(camera["target_intrinsics"], dtype=np.float32)

    pixel_frames = 1 + (latent_frames - 1) * frame_stride
    if len(world_to_camera) < pixel_frames or len(intrinsics) < pixel_frames:
        raise ValueError(
            f"Camera input needs {pixel_frames} frames for {latent_frames} latents"
        )
    camera_to_world = np.linalg.inv(world_to_camera[:pixel_frames]).astype(
        np.float32
    )
    poses = torch.from_numpy(camera_to_world).unsqueeze(0)
    calibration = torch.from_numpy(intrinsics[:pixel_frames]).unsqueeze(0)
    return poses.repeat(batch_size, 1, 1, 1), calibration.repeat(
        batch_size, 1, 1, 1
    )


class SingleImagePromptDataset(Dataset):
    """One direct image/prompt pair without a metadata wrapper."""

    def __init__(self, image_path, prompt_path, transform):
        self.image_path = Path(image_path)
        self.prompt = Path(prompt_path).read_text(encoding="utf-8").strip()
        self.transform = transform
        if not self.image_path.is_file():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        if not self.prompt:
            raise ValueError(f"Prompt is empty: {prompt_path}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        with Image.open(self.image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return {"image": image, "prompts": self.prompt, "idx": idx}


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--image_path", type=str, help="Direct I2V conditioning image")
parser.add_argument("--prompt_path", type=str, help="Direct I2V prompt text file")
parser.add_argument("--camera_path", type=str, default=None,
                    help="Camera trajectory NPZ for camera-conditioned I2V")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=None,
                    help="Number of latent output frames (defaults to the config)")
parser.add_argument("--num_frame_per_block", type=int, default=None,
                    help="Override the config's causal chunk size")
parser.add_argument("--local_attn_size", type=int, default=None,
                    help="Override model_kwargs.local_attn_size")
parser.add_argument("--camera_conditioning", action="store_true",
                    help="Enable camera conditioning on a generic config")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
args = parser.parse_args()

# Initialize distributed inference only for an actual multi-process launch.
# torchrun may set LOCAL_RANK even when WORLD_SIZE=1.
distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
if distributed_world_size > 1:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = 1
    set_seed(args.seed)

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
family_default = os.path.join(os.path.dirname(args.config_path), "default_config.yaml")
default_config_path = (
    family_default if os.path.exists(family_default)
    else "configs/default_config.yaml"
)
default_config = OmegaConf.load(default_config_path)
config = OmegaConf.merge(default_config, config)
if args.num_frame_per_block is not None:
    config.num_frame_per_block = args.num_frame_per_block
if args.local_attn_size is not None:
    config.model_kwargs.local_attn_size = args.local_attn_size
if args.camera_conditioning:
    config.camera_conditioning = True
    config.camera_frame_stride = 4
    config.camera_patch_size = 16
    config.model_kwargs.camera_conditioning = True
    config.model_kwargs.camera_patch_size = 16
    config.model_kwargs.camera_init_seed = 0
num_output_frames = args.num_output_frames or config.num_training_frames
run_i2v = args.i2v or config.i2v

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    load_generator_checkpoint(
        pipeline,
        args.checkpoint_path,
        getattr(config, "model_family", "wan"),
        args.use_ema,
    )

pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)


# Create dataset
if run_i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((config.height, config.width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    if args.image_path is not None or args.prompt_path is not None:
        if args.image_path is None or args.prompt_path is None:
            raise ValueError("Direct I2V requires both --image_path and --prompt_path")
        dataset = SingleImagePromptDataset(
            args.image_path,
            args.prompt_path,
            transform,
        )
    else:
        dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

camera_poses = None
camera_intrinsics = None
if getattr(config, "camera_conditioning", False):
    if args.camera_path is None:
        raise ValueError("Camera-conditioned inference requires --camera_path")
    camera_poses, camera_intrinsics = load_camera_inputs(
        args.camera_path,
        num_output_frames,
        int(getattr(config, "camera_frame_stride", 4)),
        args.num_samples,
    )

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if run_i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, num_output_frames - 1, *config.image_or_video_shape[2:]],
            device=device,
            dtype=torch.bfloat16,
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, num_output_frames, *config.image_or_video_shape[2:]],
            device=device,
            dtype=torch.bfloat16,
        )

    # Generate the configured latent trajectory.
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        camera_poses=camera_poses,
        camera_intrinsics=camera_intrinsics,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)
