# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Self-Forcing adapters for the official Cosmos-Predict2.5 implementation."""

import types
from typing import List, Optional

import torch

from cosmos.camera_conditioning import CAMERA_FEATURE_DIM
from huggingface_hub import hf_hub_download

from utils.scheduler import FlowMatchScheduler, SchedulerInterface
from wan.modules.vae import _video_vae


DEFAULT_MODEL_ID = "nvidia/Cosmos-Predict2.5-2B"
DEFAULT_CHECKPOINT = (
    "base/pre-trained/"
    "d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
)
DEFAULT_TEXT_ENCODER_ID = "nvidia/Cosmos-Reason1-7B"


def _mean_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.mean(dim=-1, keepdim=True)) / (
        tensor.std(dim=-1, keepdim=True) + 1e-8
    )


class CosmosTextEncoder(torch.nn.Module):
    """Cosmos-Reason1 text-only adapter producing Predict2.5 embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_ENCODER_ID,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()

    @property
    def device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        conversations = [
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a helpful assistant who will provide "
                                "prompts to an image generator."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ]
            for prompt in text_prompts
        ]
        texts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
                add_vision_id=False,
            )
            for conversation in conversations
        ]
        inputs = self.processor.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.text_encoder(
            input_ids=inputs.input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # Predict2.5 concatenates the normalized outputs of all 28 language
        # layers: 28 * 3584 = 100352 channels.
        prompt_embeds = torch.cat(
            [_mean_normalize(state) for state in outputs.hidden_states[1:]],
            dim=-1,
        )
        return {"prompt_embeds": prompt_embeds}


class CosmosVAEWrapper(torch.nn.Module):
    """Wan2.1 VAE packaged with Cosmos-Predict2.5."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_ID,
        checkpoint_filename: str = "tokenizer.pth",
    ) -> None:
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653,
            -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632,
            -0.1922, -0.9497, 0.2503, -0.2921,
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708,
            2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
            1.6382, 1.1253, 2.8251, 1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        checkpoint_path = hf_hub_download(
            repo_id=model_name,
            filename=checkpoint_filename,
        )
        self.model = _video_vae(
            pretrained_path=checkpoint_path,
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        device, dtype = pixel.device, pixel.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]
        output = [
            self.model.encode(sample.unsqueeze(0), scale).float().squeeze(0)
            for sample in pixel
        ]
        return torch.stack(output, dim=0).permute(0, 2, 1, 3, 4)

    def decode_to_pixel(
        self,
        latent: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Cached VAE decode requires batch size 1"
        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]
        decode = self.model.cached_decode if use_cache else self.model.decode
        output = [
            decode(sample.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            for sample in zs
        ]
        return torch.stack(output, dim=0).permute(0, 2, 1, 3, 4)


class CosmosDiffusionWrapper(torch.nn.Module):
    """Match Cosmos-Predict2.5 to Self-Forcing's Wan wrapper contract."""

    num_transformer_blocks = 28
    frame_seq_length = 1  # Cosmos cache positions are latent-frame indices.

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_ID,
        checkpoint_filename: str = DEFAULT_CHECKPOINT,
        timestep_shift: float = 5.0,
        is_causal: bool = False,
        local_attn_size: int = -1,
        sink_size: int = 0,
        i2v: bool = True,
        camera_conditioning: bool = False,
        camera_patch_size: int = 16,
        camera_init_seed: int = 0,
    ) -> None:
        super().__init__()
        self.is_causal = is_causal
        self.i2v = i2v
        self.local_attn_size = local_attn_size
        self.camera_conditioning = bool(camera_conditioning)
        self.camera_patch_size = int(camera_patch_size)
        if self.camera_patch_size <= 0:
            raise ValueError("camera_patch_size must be positive")
        self.uniform_timestep = not is_causal
        self._cache_max_frames = 128
        self._gradient_checkpointing = False

        self.model = self._load_model(
            model_name=model_name,
            checkpoint_filename=checkpoint_filename,
            is_causal=is_causal,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        ).eval()
        if self.camera_conditioning:
            self.model.enable_camera_conditioning(
                camera_dim=CAMERA_FEATURE_DIM * self.camera_patch_size**2,
                init_seed=int(camera_init_seed),
            )

        if is_causal:
            self._kv_attention_ops = [
                block.self_attn.attn_op for block in self.model.blocks
            ]
            for attention_op in self._kv_attention_ops:
                attention_op.reset_kv_cache(
                    max_cache_size=self._cache_max_frames,
                )

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.post_init()

    @staticmethod
    def _model_kwargs(is_causal: bool) -> dict:
        from cosmos.minimal_v4_dit import SACConfig

        return {
            "max_img_h": 240,
            "max_img_w": 240,
            "max_frames": 128,
            "in_channels": 16,
            "out_channels": 16,
            "patch_spatial": 2,
            "patch_temporal": 1,
            "model_channels": 2048,
            "num_blocks": 28,
            "num_heads": 16,
            "concat_padding_mask": True,
            "pos_emb_cls": "rope3d",
            "pos_emb_learnable": True,
            "pos_emb_interpolation": "crop",
            "use_adaln_lora": True,
            "adaln_lora_dim": 256,
            "extra_per_block_abs_pos_emb": False,
            "rope_enable_fps_modulation": False,
            "rope_h_extrapolation_ratio": 3.0,
            "rope_w_extrapolation_ratio": 3.0,
            "rope_t_extrapolation_ratio": 1.0,
            "use_crossattn_projection": True,
            "crossattn_proj_in_channels": 100352,
            "crossattn_emb_channels": 1024,
            "timestep_scale": 0.001,
            "use_wan_fp32_strategy": True,
            "atten_backend": "i4" if is_causal else "minimal_a2a",
            "sac_config": SACConfig(mode="none"),
        }

    @classmethod
    def _load_model(
        cls,
        model_name: str,
        checkpoint_filename: str,
        is_causal: bool,
        local_attn_size: int,
        sink_size: int,
    ) -> torch.nn.Module:
        if is_causal:
            from cosmos.causal_model import CausalCosmosModel as Model
        else:
            from cosmos.minimal_v1_lvg_dit import (
                MinimalV1LVGDiT as Model,
            )

        with torch.device("meta"):
            model_kwargs = cls._model_kwargs(is_causal)
            if is_causal:
                model_kwargs.update(
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                )
            model = Model(**model_kwargs)

        checkpoint_path = hf_hub_download(
            repo_id=model_name,
            filename=checkpoint_filename,
        )
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        available_prefixes = (
            "net_ema." if any(key.startswith("net_ema.") for key in state_dict)
            else "net."
        )
        net_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith(available_prefixes):
                continue
            key = key.removeprefix(available_prefixes)
            if not key.endswith("_extra_state"):
                net_state_dict[key] = value

        if not net_state_dict:
            raise RuntimeError("Cosmos checkpoint contains no diffusion weights")

        model.load_state_dict(net_state_dict, strict=False, assign=True)
        unloaded = [
            name
            for name, tensor in (
                list(model.named_parameters()) + list(model.named_buffers())
            )
            if tensor.is_meta
        ]
        if unloaded:
            raise RuntimeError(
                "Cosmos checkpoint did not initialize parameters: "
                + ", ".join(unloaded[:10])
            )
        return model

    def enable_gradient_checkpointing(self) -> None:
        if self._gradient_checkpointing:
            return
        from cosmos.minimal_v4_dit import SACConfig

        # FlexAttention is a higher-order op and PyTorch does not implement it
        # for selective checkpointing's _CachingTorchDispatchMode. Ordinary
        # block checkpointing preserves the memory saving without that mode.
        self.model.enable_selective_checkpoint(
            SACConfig(mode="block_wise"),
            self.model.blocks,
        )
        self._gradient_checkpointing = True

    def initialize_kv_cache(
        self,
        max_frames: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> object:
        del batch_size, dtype, device
        if not self.is_causal:
            raise RuntimeError("KV cache is only available on the causal Cosmos model")
        self._cache_max_frames = max_frames
        for attention_op in self._kv_attention_ops:
            attention_op.reset_kv_cache(max_cache_size=max_frames)
        return self  # The actual cache is owned by each attention block.

    def initialize_crossattn_cache(self, **kwargs) -> None:
        del kwargs
        return None

    @staticmethod
    def cache_position(frame_index: int) -> int:
        return frame_index

    def _condition_inputs(
        self,
        noisy_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        apply_initial_condition: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames, _, height, width = noisy_video.shape
        mask = torch.zeros(
            batch,
            1,
            frames,
            height,
            width,
            device=noisy_video.device,
            dtype=noisy_video.dtype,
        )
        initial_latent = conditional_dict.get("initial_latent")
        if self.i2v and initial_latent is not None and apply_initial_condition:
            cond_frames = min(initial_latent.shape[1], frames)
            noisy_video = noisy_video.clone()
            noisy_video[:, :cond_frames] = initial_latent[:, :cond_frames]
            mask[:, :, :cond_frames] = 1
            timestep = timestep.clone()
            timestep[:, :cond_frames] = 0
        return noisy_video, timestep, mask

    def _camera_condition(
        self,
        conditional_dict: dict,
        model_input: torch.Tensor,
        *,
        current_start: Optional[int],
        streaming: bool,
    ) -> Optional[torch.Tensor]:
        camera = conditional_dict.get("camera_condition")
        if not self.camera_conditioning:
            if camera is not None:
                raise ValueError(
                    "camera_condition was provided, but camera_conditioning is disabled"
                )
            return None
        if camera is None:
            raise ValueError(
                "camera_conditioning is enabled, but conditional_dict has no camera_condition"
            )
        if camera.ndim != 5:
            raise ValueError(
                "camera_condition must have shape [B, C, T, H, W]; got "
                f"{tuple(camera.shape)}"
            )

        target_frames = model_input.shape[1]
        start = int(current_start or 0) if streaming else 0
        if camera.shape[2] >= start + target_frames:
            camera = camera[:, :, start : start + target_frames]
        elif camera.shape[2] != target_frames:
            raise ValueError(
                "Camera conditioning is too short for the requested video frames: "
                f"start={start}, frames={target_frames}, camera_frames={camera.shape[2]}"
            )

        expected_spatial_grid = (
            model_input.shape[-2] // self.model.patch_spatial,
            model_input.shape[-1] // self.model.patch_spatial,
        )
        if camera.shape[-2:] != expected_spatial_grid:
            raise ValueError(
                "Camera and video spatial token grids do not match: camera="
                f"{tuple(camera.shape[-2:])}, video="
                f"{expected_spatial_grid}"
            )
        if camera.shape[0] != model_input.shape[0]:
            if model_input.shape[0] % camera.shape[0]:
                raise ValueError(
                    "Camera batch cannot be expanded to the model batch: "
                    f"{camera.shape[0]} and {model_input.shape[0]}"
                )
            camera = camera.repeat(
                model_input.shape[0] // camera.shape[0], 1, 1, 1, 1
            )
        return camera

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[object] = None,
        crossattn_cache: Optional[object] = None,
        current_start: Optional[int] = None,
        classify_mode: bool = False,
        concat_time_embeddings: bool = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        store_kv: Optional[bool] = None,
    ) -> torch.Tensor:
        del crossattn_cache, concat_time_embeddings, cache_start
        if classify_mode:
            raise NotImplementedError("Cosmos GAN classifier branch is not integrated")
        teacher_forcing = clean_x is not None or aug_t is not None
        if teacher_forcing and (clean_x is None or aug_t is None):
            raise ValueError("Teacher forcing requires both clean_x and aug_t")
        if teacher_forcing and (not self.is_causal or kv_cache is not None):
            raise ValueError(
                "Teacher forcing requires a causal model without a streaming KV cache"
            )

        prompt_embeds = conditional_dict["prompt_embeds"]
        if self.uniform_timestep:
            timestep = timestep[:, :1].expand(-1, noisy_image_or_video.shape[1])
        model_input, input_timestep, condition_mask = self._condition_inputs(
            noisy_image_or_video,
            conditional_dict,
            timestep,
            # Full-sequence training conditions frame zero. Streaming inference
            # has already cached that frame, so later chunks must remain noisy.
            apply_initial_condition=(
                kv_cache is None or int(current_start or 0) == 0
            ),
        )
        clean_model_input = None
        clean_input_timestep = None
        if teacher_forcing:
            clean_model_input, clean_input_timestep, clean_condition_mask = (
                self._condition_inputs(
                    clean_x,
                    conditional_dict,
                    aug_t,
                    apply_initial_condition=True,
                )
            )
            if clean_condition_mask.shape != condition_mask.shape:
                raise ValueError(
                    "Clean and noisy condition masks must have matching shapes"
                )
        camera_condition = self._camera_condition(
            conditional_dict,
            model_input,
            current_start=current_start,
            streaming=kv_cache is not None,
        )
        # Keep flow/noise/target construction in FP32, then cast only at the
        # DiT boundary just like a mixed-precision root module would.
        compute_dtype = self.model.x_embedder.proj[1].weight.dtype
        model_input_bcthw = model_input.to(dtype=compute_dtype).permute(0, 2, 1, 3, 4)
        clean_model_input_bcthw = (
            clean_model_input.to(dtype=compute_dtype).permute(0, 2, 1, 3, 4)
            if clean_model_input is not None
            else None
        )
        prompt_embeds = prompt_embeds.to(dtype=compute_dtype)
        condition_mask = condition_mask.to(dtype=compute_dtype)
        if camera_condition is not None:
            camera_condition = camera_condition.to(
                device=model_input.device,
                dtype=compute_dtype,
            )
        padding_mask = torch.zeros(
            model_input.shape[0],
            1,
            model_input.shape[-2],
            model_input.shape[-1],
            device=model_input.device,
            dtype=compute_dtype,
        )

        if kv_cache is not None:
            from cosmos.kv_cache import (
                KVCacheConfig,
                VideoSeqPos,
            )

            if model_input.shape[1] != 1:
                raise ValueError(
                    "Cosmos causal rollout uses one latent frame per block; "
                    "set num_frame_per_block: 1"
                )
            frame_index = int(current_start or 0)
            token_h = model_input.shape[-2] // self.model.patch_spatial
            token_w = model_input.shape[-1] // self.model.patch_spatial
            video_pos = VideoSeqPos(
                T=self._cache_max_frames,
                H=token_h,
                W=token_w,
            ).frame(frame_index)
            should_store_kv = (
                bool(torch.all(input_timestep == 0).item())
                if store_kv is None else bool(store_kv)
            )
            flow_pred = self.model.forward_seq(
                x_B_C_T_H_W=model_input_bcthw,
                video_pos=video_pos,
                timesteps_B_T=input_timestep,
                crossattn_emb=prompt_embeds,
                padding_mask=padding_mask,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                camera_condition_B_C_T_H_W=camera_condition,
                kv_cache_cfg=KVCacheConfig(
                    run_with_kv=True,
                    store_kv=should_store_kv,
                    current_idx=frame_index,
                ),
            ).permute(0, 2, 1, 3, 4)
        elif teacher_forcing:
            flow_pred = self.model.forward_teacher_forcing(
                noisy_x_B_C_T_H_W=model_input_bcthw,
                clean_x_B_C_T_H_W=clean_model_input_bcthw,
                noisy_timesteps_B_T=input_timestep,
                clean_timesteps_B_T=clean_input_timestep,
                crossattn_emb=prompt_embeds,
                padding_mask=padding_mask,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                camera_condition_B_C_T_H_W=camera_condition,
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                x_B_C_T_H_W=model_input_bcthw,
                timesteps_B_T=input_timestep,
                crossattn_emb=prompt_embeds,
                padding_mask=padding_mask,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                camera_condition_B_C_T_H_W=camera_condition,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=model_input.flatten(0, 1),
            timestep=input_timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0

    def _convert_flow_pred_to_x0(
        self,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda value: value.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return (xt - sigma_t * flow_pred).to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler,
        x0_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda value: value.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return ((xt - x0_pred) / sigma_t).to(original_dtype)

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise,
            scheduler,
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0,
            scheduler,
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0,
            scheduler,
        )
        return scheduler

    def post_init(self) -> None:
        self.get_scheduler()
