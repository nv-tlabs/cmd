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

import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import (
    EMA_FSDP,
    create_fsdp2_device_mesh,
    distributed_clip_grad_norm_,
    fsdp2_wrap_cosmos_model,
    fsdp_state_dict,
    fsdp_wrap,
    launch_distributed_job,
)
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, ContextMatchedDistillation, DMD, SiD
import torch
import wandb
import time
import os

from cosmos.camera_conditioning import build_camera_conditioning


def load_checkpoint(path):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu")


def save_safetensors_checkpoint(state_dict, path, *, component, step):
    """Save one flat model component without pickle serialization."""
    from safetensors.torch import save_file

    tensors = {}
    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            raise TypeError(
                f"Safetensors checkpoint entry {name!r} is not a tensor: "
                f"{type(value).__name__}"
            )
        tensors[name] = value.detach().cpu().contiguous()

    save_file(
        tensors,
        path,
        metadata={
            "component": component,
            "step": str(step),
        },
    )


_COSMOS_CHECKPOINT_OPTIONAL_BUFFERS = {
    "model.accum_video_sample_counter",
    "model.accum_image_sample_counter",
    "model.accum_iteration",
    "model.accum_train_in_hours",
}


def load_model_state_dict(module, state_dict, model_family, description):
    """Load exported Cosmos DiT weights into the diffusion-wrapper FSDP module."""
    if model_family != "cosmos":
        module.load_state_dict(state_dict, strict=True)
        return

    # Exported Cosmos safetensors contain bare DiT names (`blocks.*`), while
    # the training object is a diffusion wrapper whose DiT is named `model`.
    normalized = {}
    for name, value in state_dict.items():
        normalized_name = name if name.startswith("model.") else f"model.{name}"
        if normalized_name in normalized:
            raise RuntimeError(
                f"Duplicate key after Cosmos checkpoint normalization: {normalized_name}"
            )
        normalized[normalized_name] = value

    incompatible = module.load_state_dict(normalized, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    invalid_missing = missing - _COSMOS_CHECKPOINT_OPTIONAL_BUFFERS
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Incompatible {description} Cosmos checkpoint after key normalization: "
            f"missing={sorted(invalid_missing)}, unexpected={sorted(unexpected)}"
        )
    if dist.get_rank() == 0:
        print(
            f"Loaded {description}: {len(normalized)} tensors; "
            f"kept {len(missing)} runtime bookkeeping buffers",
            flush=True,
        )


def load_pretrained_components(model, config):
    """Load the configured generator and score checkpoints."""
    state_dict = None
    loaded_checkpoint = None
    if getattr(config, "generator_ckpt", False):
        print(f"Loading pretrained generator from {config.generator_ckpt}")
        state_dict = load_checkpoint(config.generator_ckpt)
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        load_model_state_dict(
            model.generator,
            state_dict,
            getattr(config, "model_family", "wan"),
            "generator",
        )
        loaded_checkpoint = config.generator_ckpt

    if getattr(config, "teacher_ckpt", False):
        if config.teacher_ckpt != loaded_checkpoint:
            state_dict = load_checkpoint(config.teacher_ckpt)
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
        print(f"Loading causal real and fake scores from {config.teacher_ckpt}")
        model_family = getattr(config, "model_family", "wan")
        load_model_state_dict(
            model.real_score,
            state_dict,
            model_family,
            "real score",
        )
        load_model_state_dict(
            model.fake_score,
            state_dict,
            model_family,
            "fake score",
        )


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "context_matched":
            self.model = ContextMatchedDistillation(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        use_fsdp2_scores = (
            getattr(config, "model_family", "wan") == "cosmos"
            and config.distribution_loss == "context_matched"
        )
        if use_fsdp2_scores:
            # FSDP2 parameters become DTensors, so load ordinary full tensors
            # before applying composable sharding.
            load_pretrained_components(self.model, config)

        if use_fsdp2_scores:
            score_mesh = create_fsdp2_device_mesh()
            self.model.generator = fsdp2_wrap_cosmos_model(
                self.model.generator,
                mesh=score_mesh,
                mixed_precision=config.mixed_precision,
            )
            self.model.real_score = fsdp2_wrap_cosmos_model(
                self.model.real_score,
                mesh=score_mesh,
                mixed_precision=config.mixed_precision,
            )
            self.model.fake_score = fsdp2_wrap_cosmos_model(
                self.model.fake_score,
                mesh=score_mesh,
                mixed_precision=config.mixed_precision,
            )
            if self.is_main_process:
                print(
                    "Using explicit block-sharded FSDP2 for Cosmos models",
                    flush=True,
                )
        else:
            self.model.generator = fsdp_wrap(
                self.model.generator,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.generator_fsdp_wrap_strategy
            )
            self.model.real_score = fsdp_wrap(
                self.model.real_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.real_score_fsdp_wrap_strategy
            )

            self.model.fake_score = fsdp_wrap(
                self.model.fake_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.fake_score_fsdp_wrap_strategy
            )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not use_fsdp2_scores:
            load_pretrained_components(self.model, config)

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(
                config.data_path,
                max_pair=int(1e8),
                load_camera=bool(
                    getattr(config, "camera_conditioning", False)
                ),
            )
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader_num_workers = int(
            getattr(config, "dataloader_num_workers", 8)
        )
        if dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=dataloader_num_workers)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)
        self.validation_callback = None
        if (
            getattr(config, "model_family", "wan") == "cosmos"
            and int(getattr(config, "validation_interval", 0)) > 0
        ):
            from cosmos.validation_callback import CosmosValidationCallback

            self.validation_callback = CosmosValidationCallback(config)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.grad_accum_iter = int(getattr(config, "grad_accum_iter", 1))
        if self.grad_accum_iter <= 0:
            raise ValueError("grad_accum_iter must be positive")
        self.rollout_num_chunks = int(getattr(config, "rollout_num_chunks", 1))
        if self.rollout_num_chunks > 1 and self.grad_accum_iter != 1:
            raise ValueError("Rollout training requires grad_accum_iter: 1")
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.generator_ema is not None:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path,
                f"checkpoint_model_{self.step:06d}",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            training_checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, training_checkpoint_path)

            generator_path = os.path.join(checkpoint_dir, "model.safetensors")
            save_safetensors_checkpoint(
                generator_state_dict,
                generator_path,
                component="generator",
                step=self.step,
            )
            print(
                "Training checkpoint and generator safetensors saved to",
                training_checkpoint_path,
                generator_path,
            )

    def fwdbwd_one_step(
        self, batch, train_generator, loss_scale=1.0,
        rollout_chunk_index=None,
    ):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = batch["ode_latent"][:, -1].to(
                device=self.device, dtype=self.dtype
            ) if rollout_chunk_index is not None else None
            image_latent = batch["ode_latent"][:, -1][:, 0:1].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            # Classifier-free guidance must keep the same input image for both
            # text branches; only the prompt is unconditional.
            if self.config.i2v:
                conditional_dict["initial_latent"] = image_latent
                unconditional_dict["initial_latent"] = image_latent

            if getattr(self.config, "camera_conditioning", False):
                if "camera_poses" not in batch or "camera_intrinsics" not in batch:
                    raise RuntimeError(
                        "Camera-conditioned distillation requires camera_poses "
                        "and camera_intrinsics in every batch"
                    )
                camera_condition = build_camera_conditioning(
                    batch["camera_poses"].to(
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    batch["camera_intrinsics"].to(
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    image_height=int(self.config.height),
                    image_width=int(self.config.width),
                    frame_stride=int(
                        getattr(self.config, "camera_frame_stride", 4)
                    ),
                    patch_size=int(
                        getattr(self.config, "camera_patch_size", 16)
                    ),
                    num_frame_per_block=int(
                        getattr(self.config, "num_frame_per_block", 1)
                    ),
                    expected_latent_frames=image_or_video_shape[1],
                    output_dtype=self.dtype,
                )
                conditional_dict["camera_condition"] = camera_condition
                unconditional_dict["camera_condition"] = camera_condition

        # Step 3: Store gradients for the generator (if training the generator)
        rollout_kwargs = (
            {"rollout_chunk_index": rollout_chunk_index}
            if rollout_chunk_index is not None else {}
        )
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                **rollout_kwargs,
            )

            (generator_loss * loss_scale).backward()

            generator_log_dict.update(
                {"generator_loss": generator_loss.detach()}
            )

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None,
            **rollout_kwargs,
        )

        (critic_loss * loss_scale).backward()

        critic_log_dict.update({"critic_loss": critic_loss.detach()})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [
                    batch_size,
                    self.model.num_training_frames - 1,
                    *self.config.image_or_video_shape[2:],
                ],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [
                    batch_size,
                    self.model.num_training_frames,
                    *self.config.image_or_video_shape[2:],
                ],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step
        validate_first_batch = bool(
            self.validation_callback is not None
            and getattr(self.config, "validation_at_start", False)
        )

        max_steps = getattr(self.config, "max_steps", None)
        rollout_chunk = 0
        generator_rollout_batch = None
        critic_rollout_batch = None
        while max_steps is None or self.step < max_steps:
            validation_ran = False
            rollout_step = self.step // self.rollout_num_chunks
            TRAIN_GENERATOR = (
                rollout_step % self.config.dfake_gen_update_ratio == 0
            )
            first_iteration = self.step == start_step

            # Train the generator
            if TRAIN_GENERATOR:
                if self.is_main_process and first_iteration:
                    print(
                        f"Starting generator forward/backward at step {self.step}",
                        flush=True,
                    )
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                for _ in range(self.grad_accum_iter):
                    if self.rollout_num_chunks > 1:
                        if rollout_chunk == 0:
                            generator_rollout_batch = next(self.dataloader)
                        batch = generator_rollout_batch
                    else:
                        batch = next(self.dataloader)
                    if validate_first_batch:
                        self.validation_callback.run(self, batch)
                        validate_first_batch = False
                        validation_ran = True
                    extra = self.fwdbwd_one_step(
                        batch,
                        True,
                        loss_scale=1.0 / self.grad_accum_iter,
                        rollout_chunk_index=(
                            rollout_chunk if self.rollout_num_chunks > 1 else None
                        ),
                    )
                    extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                generator_log_dict["generator_grad_norm"] = (
                    distributed_clip_grad_norm_(
                        self.model.generator,
                        self.max_grad_norm_generator
                    )
                )
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                if self.is_main_process and first_iteration:
                    print(
                        f"Finished generator update at step {self.step}",
                        flush=True,
                    )

            else:
                if self.is_main_process and first_iteration:
                    print(
                        f"Starting critic forward/backward at step {self.step}",
                        flush=True,
                    )
                self.critic_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                for _ in range(self.grad_accum_iter):
                    if self.rollout_num_chunks > 1:
                        if rollout_chunk == 0:
                            critic_rollout_batch = next(self.dataloader)
                        batch = critic_rollout_batch
                    else:
                        batch = next(self.dataloader)
                    extra = self.fwdbwd_one_step(
                        batch,
                        False,
                        loss_scale=1.0 / self.grad_accum_iter,
                        rollout_chunk_index=(
                            rollout_chunk
                            if self.rollout_num_chunks > 1 else None
                        ),
                    )
                    extras_list.append(extra)
                critic_log_dict = merge_dict_list(extras_list)
                critic_log_dict["critic_grad_norm"] = (
                    distributed_clip_grad_norm_(
                        self.model.fake_score,
                        self.max_grad_norm_critic
                    )
                )
                self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1
            if self.rollout_num_chunks > 1:
                rollout_chunk = (rollout_chunk + 1) % self.rollout_num_chunks
                if rollout_chunk == 0:
                    self.model.reset_rollout_state()
                    generator_rollout_batch = None
                    critic_rollout_batch = None
            if self.is_main_process and first_iteration:
                print(f"Finished training step {self.step}", flush=True)

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                else:
                    wandb_loss_dict.update(
                        {
                            "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                            "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                        }
                    )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if (
                self.validation_callback is not None
                and self.validation_callback.should_run(self.step)
            ):
                self.validation_callback.run(self, batch)
                validation_ran = True

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None or validation_ran:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
