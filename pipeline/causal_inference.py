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

from typing import List, Optional
import torch

from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.model_factory import (
    build_diffusion_wrapper,
    build_text_encoder,
    build_vae,
)
from cosmos.camera_conditioning import build_camera_conditioning


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = (
            build_diffusion_wrapper(args, is_causal=True)
            if generator is None else generator
        )
        self.text_encoder = build_text_encoder(args) if text_encoder is None else text_encoder
        self.vae = build_vae(args) if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.num_inference_steps = int(getattr(args, "num_inference_steps", 0))
        self.denoising_step_list = None
        if self.num_inference_steps <= 0:
            self.denoising_step_list = torch.tensor(
                args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = getattr(self.generator, "num_transformer_blocks", 30)
        self.frame_seq_length = getattr(self.generator, "frame_seq_length", 1560)

        self.kv_cache1 = None
        self.args = args
        self.context_noise = int(getattr(args, "context_noise", 0))
        if self.context_noise < 0:
            raise ValueError("context_noise must be non-negative")
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = getattr(self.generator.model, "local_attn_size", -1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1 and hasattr(self.generator.model, "num_frame_per_block"):
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        camera_poses: Optional[torch.Tensor] = None,
        camera_intrinsics: Optional[torch.Tensor] = None,
        camera_condition: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        decode: bool = True,
        profile: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = None
        if self.num_inference_steps > 0 and self.args.guidance_scale > 1.0:
            unconditional_dict = self.text_encoder(
                text_prompts=[self.args.negative_prompt] * len(text_prompts)
            )
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
            if unconditional_dict is not None:
                unconditional_dict["initial_latent"] = initial_latent
        if getattr(self.args, "camera_conditioning", False):
            if camera_condition is None:
                if camera_poses is None or camera_intrinsics is None:
                    raise ValueError(
                        "Camera-conditioned inference requires camera_poses and "
                        "camera_intrinsics, or a precomputed camera_condition"
                    )
                camera_condition = build_camera_conditioning(
                    camera_poses.to(device=noise.device, dtype=torch.float32),
                    camera_intrinsics.to(device=noise.device, dtype=torch.float32),
                    image_height=int(self.args.height),
                    image_width=int(self.args.width),
                    frame_stride=int(getattr(self.args, "camera_frame_stride", 4)),
                    patch_size=int(getattr(self.args, "camera_patch_size", 16)),
                    expected_latent_frames=num_output_frames,
                    output_dtype=noise.dtype,
                )
            conditional_dict["camera_condition"] = camera_condition
            if unconditional_dict is not None:
                unconditional_dict["camera_condition"] = camera_condition

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if hasattr(self.generator, "initialize_kv_cache"):
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                max_frames=num_output_frames,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
            )
        elif self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=self._cache_position(current_start_frame),
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=self._cache_position(current_start_frame),
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Integrate the flow trajectory for a regular FM model.
            # The legacy x0/re-noise path below is retained only for distilled
            # few-step configs that explicitly provide denoising_step_list.
            if self.num_inference_steps > 0:
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.args.num_train_timestep,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(
                    self.num_inference_steps,
                    device=noise.device,
                    shift=self.args.timestep_shift,
                )
                denoised_pred = noisy_input
                for current_timestep in sample_scheduler.timesteps:
                    if profile:
                        print(f"current_timestep: {current_timestep}")
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=current_timestep.dtype,
                    ) * current_timestep
                    if unconditional_dict is not None:
                        cfg_dict = dict(conditional_dict)
                        cfg_dict["prompt_embeds"] = torch.cat(
                            [
                                conditional_dict["prompt_embeds"],
                                unconditional_dict["prompt_embeds"],
                            ],
                            dim=0,
                        )
                        flow_pred_cfg, _ = self.generator(
                            noisy_image_or_video=torch.cat(
                                [denoised_pred, denoised_pred], dim=0
                            ),
                            conditional_dict=cfg_dict,
                            timestep=torch.cat([timestep, timestep], dim=0),
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=self._cache_position(current_start_frame),
                        )
                        flow_pred_cond, flow_pred_uncond = flow_pred_cfg.chunk(2)
                        # Cosmos Predict2.5 V2W guidance is applied in velocity
                        # space around the conditional prediction.
                        flow_pred = flow_pred_cond + self.args.guidance_scale * (
                            flow_pred_cond - flow_pred_uncond
                        )
                    else:
                        flow_pred, _ = self.generator(
                            noisy_image_or_video=denoised_pred,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=self._cache_position(current_start_frame),
                        )
                    denoised_pred = sample_scheduler.step(
                        flow_pred,
                        current_timestep,
                        denoised_pred,
                        return_dict=False,
                    )[0]
            else:
                for index, current_timestep in enumerate(self.denoising_step_list):
                    if profile:
                        print(f"current_timestep: {current_timestep}")
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if index < len(self.denoising_step_list) - 1:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=self._cache_position(current_start_frame)
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=self._cache_position(current_start_frame)
                        )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: commit the generated frame to K/V at the context-noise
            # level used by the causal training/inference recipe. The original
            # I2V prefix remains clean at timestep zero.
            context_timestep = torch.ones_like(timestep) * self.context_noise
            cache_input = denoised_pred.detach()
            if self.context_noise > 0:
                cache_input = self.scheduler.add_noise(
                    cache_input.flatten(0, 1),
                    torch.randn_like(cache_input.flatten(0, 1)),
                    context_timestep.flatten(0, 1),
                ).unflatten(0, cache_input.shape[:2])
            self.generator(
                noisy_image_or_video=cache_input,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=self._cache_position(current_start_frame),
                store_kv=True,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output when pixels are requested.
        video = None
        if decode:
            video = self.vae.decode_to_pixel(output, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end) if decode else 0.0
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        if not decode:
            return output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, max_frames=None):
        """
        Initialize the backend's causal KV cache.
        """
        if hasattr(self.generator, "initialize_kv_cache"):
            self.kv_cache1 = self.generator.initialize_kv_cache(
                max_frames=max_frames or getattr(self.args, "num_training_frames", 21),
                batch_size=batch_size,
                dtype=dtype,
                device=device,
            )
            return

        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize the backend's cross-attention cache when it has one.
        """
        if hasattr(self.generator, "initialize_crossattn_cache"):
            self.crossattn_cache = self.generator.initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=dtype,
                device=device,
            )
            return

        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _cache_position(self, frame_index: int) -> int:
        if hasattr(self.generator, "cache_position"):
            return self.generator.cache_position(frame_index)
        return frame_index * self.frame_seq_length
