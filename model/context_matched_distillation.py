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

from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from model.base import SelfForcingModel


class ContextMatchedDistillation(SelfForcingModel):
    def __init__(self, args, device):
        """Initialize causal scores conditioned on expanding student context."""
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)
        self.context_matched_distillation = (
            getattr(args, "distribution_loss", None) == "context_matched"
        )
        if not self.context_matched_distillation:
            raise ValueError(
                "ContextMatchedDistillation requires distribution_loss: context_matched"
        )
        self.prefix_noise = int(getattr(args, "prefix_noise", 0))
        self.rollout_num_chunks = int(getattr(args, "rollout_num_chunks", 1))
        self.score_context_noise_boundaries = list(
            getattr(args, "score_context_noise_frame_boundaries", []) or []
        )
        self.score_context_noise_values = list(
            getattr(args, "score_context_noise_frame_values", []) or []
        )
        if self.score_context_noise_boundaries and len(
            self.score_context_noise_values
        ) != len(self.score_context_noise_boundaries) + 1:
            raise ValueError("Invalid score context noise schedule")
        self.rollout_mean_weight = float(
            getattr(args, "rollout_first_gt_latent_mean_loss_weight", 0.0)
        )
        self.rollout_std_weight = float(
            getattr(args, "rollout_first_gt_latent_std_loss_weight", 0.0)
        )
        self._rollout_states = {}

        if self.context_matched_distillation:
            if getattr(args, "model_family", "wan") != "cosmos":
                raise ValueError(
                    "Context-Matched Distillation currently requires Cosmos"
                )
            if not args.i2v or not args.independent_first_frame:
                raise ValueError(
                    "Context-Matched Distillation requires independent-first-frame I2V"
                )
            if self.num_training_frames < int(args.image_or_video_shape[1]):
                raise ValueError(
                    "Context-Matched Distillation requires num_training_frames "
                    "to cover the "
                    "scored image_or_video_shape window"
                )
            if self.prefix_noise < 0:
                raise ValueError("prefix_noise must be non-negative")
            rollout_frames = (
                self.rollout_num_chunks * int(args.image_or_video_shape[1])
            )
            if self.rollout_num_chunks > 1 and (
                self.num_frame_per_block != 1
                or rollout_frames > self.num_training_frames
                or rollout_frames > 128
            ):
                raise ValueError("Invalid rollout geometry")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if self.context_matched_distillation:
            for score_model in (self.real_score, self.fake_score):
                score_model.model.num_frame_per_block = self.num_frame_per_block
                score_model.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _split_context_matched_rollout(
        self,
        image_or_video: torch.Tensor,
        gradient_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Split the clean I2V frame from the 23 generated/scored frames."""
        scored_frames = int(self.args.image_or_video_shape[1]) - 1
        if image_or_video.shape[1] < scored_frames + 1:
            raise ValueError(
                "Context-Matched Distillation received too few generated frames"
            )
        suffix_start = image_or_video.shape[1] - scored_frames
        prefix = image_or_video[:, 1:suffix_start].detach()
        scored_suffix = image_or_video[:, suffix_start:]
        if gradient_mask is not None:
            gradient_mask = gradient_mask[:, suffix_start:]
        return scored_suffix, prefix, gradient_mask

    def _prefix_timesteps(self, prefix: torch.Tensor, start: int) -> torch.Tensor:
        if not self.score_context_noise_boundaries:
            return torch.full(
                prefix.shape[:2], self.prefix_noise,
                device=prefix.device, dtype=torch.float32,
            )
        indices = torch.arange(
            start, start + prefix.shape[1], device=prefix.device
        )
        buckets = torch.bucketize(
            indices,
            torch.tensor(
                self.score_context_noise_boundaries, device=prefix.device
            ),
            right=True,
        )
        values = torch.tensor(
            self.score_context_noise_values,
            device=prefix.device,
            dtype=torch.float32,
        )
        return values[buckets].expand(prefix.shape[0], -1)

    def _noise_prefix(self, prefix: torch.Tensor, start: int):
        prefix_timestep = self._prefix_timesteps(prefix, start)
        if prefix.shape[1] == 0:
            return prefix, prefix_timestep
        noised_prefix = self.scheduler.add_noise(
            prefix.flatten(0, 1),
            torch.randn_like(prefix.flatten(0, 1)),
            prefix_timestep.flatten(0, 1),
        ).unflatten(0, prefix.shape[:2])
        return noised_prefix, prefix_timestep

    def _score_with_prefix(
        self,
        score_model: torch.nn.Module,
        noisy_suffix: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        noised_history: Optional[torch.Tensor],
        history_timestep: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score the suffix with history and noisy targets packed together."""
        if not self.context_matched_distillation:
            return score_model(
                noisy_image_or_video=noisy_suffix,
                conditional_dict=conditional_dict,
                timestep=timestep,
            )

        initial_latent = conditional_dict.get("initial_latent")
        if initial_latent is None or initial_latent.shape[1] != 1:
            raise ValueError(
                "Context-Matched Distillation requires one initial I2V latent"
            )
        if noised_history is None:
            raise ValueError(
                "Context-Matched Distillation requires noised student history"
            )
        if history_timestep is None:
            raise ValueError("Context-Matched Distillation requires history timesteps")

        initial_latent = initial_latent.to(
            device=noisy_suffix.device,
            dtype=noisy_suffix.dtype,
        )
        context = torch.cat(
            [initial_latent, noised_history],
            dim=1,
        )
        context_timestep = torch.cat(
            [
                torch.zeros_like(timestep[:, :1]),
                history_timestep.to(timestep.dtype),
            ],
            dim=1,
        )
        suffix_start = context.shape[1]
        flow_pred, x0_pred = score_model(
            noisy_image_or_video=torch.cat([context, noisy_suffix], dim=1),
            conditional_dict=conditional_dict,
            timestep=torch.cat([context_timestep, timestep], dim=1),
            clean_x=context,
            aug_t=context_timestep,
            teacher_forcing_start=suffix_start,
        )
        return (
            flow_pred[:, suffix_start:],
            x0_pred[:, suffix_start:],
        )

    def _score_cfg_pair(
        self,
        score_model,
        noisy_image_or_video,
        timestep,
        conditional_dict,
        unconditional_dict,
        noised_history,
        history_timestep,
    ):
        batch_size = noisy_image_or_video.shape[0]
        conditioning = {
            key: torch.cat([value, unconditional_dict[key]], dim=0)
            for key, value in conditional_dict.items()
        }
        _, prediction = self._score_with_prefix(
            score_model,
            torch.cat([noisy_image_or_video, noisy_image_or_video], dim=0),
            torch.cat([timestep, timestep], dim=0),
            conditioning,
            torch.cat([noised_history, noised_history], dim=0),
            torch.cat([history_timestep, history_timestep], dim=0),
        )
        return prediction[:batch_size], prediction[batch_size:]

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        noised_history: Optional[torch.Tensor] = None,
        history_timestep: Optional[torch.Tensor] = None,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        if self.fake_guidance_scale != 0.0:
            pred_fake_image_cond, pred_fake_image_uncond = self._score_cfg_pair(
                self.fake_score,
                noisy_image_or_video,
                timestep,
                conditional_dict,
                unconditional_dict,
                noised_history,
                history_timestep,
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            _, pred_fake_image = self._score_with_prefix(
                self.fake_score,
                noisy_image_or_video,
                timestep,
                conditional_dict,
                noised_history,
                history_timestep,
            )

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        pred_real_image_cond, pred_real_image_uncond = self._score_cfg_pair(
            self.real_score,
            noisy_image_or_video,
            timestep,
            conditional_dict,
            unconditional_dict,
            noised_history,
            history_timestep,
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        prefix: Optional[torch.Tensor] = None,
        prefix_start: int = 1,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True,
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))
            noised_history, history_timestep = self._noise_prefix(
                prefix, prefix_start
            )

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                noised_history=noised_history,
                history_timestep=history_timestep,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def reset_rollout_state(self):
        self._rollout_states.clear()

    def _run_rollout_generator(
        self,
        image_or_video_shape,
        conditional_dict,
        initial_latent,
        chunk_index,
        phase,
    ):
        if chunk_index == 0:
            self._rollout_states[phase] = {"history": None, "exit_step": None}
        state = self._rollout_states[phase]
        history = state["history"]
        if history is None:
            prefix = initial_latent[:, :0]
            prefix_start = 1
            context = None
            chunk_initial = initial_latent
            generated_frames = image_or_video_shape[1] - 1
        else:
            available = history[:, 1:]
            window_frames = int(image_or_video_shape[1]) - 1
            prefix = available[:, -window_frames:].detach()
            prefix_start = history.shape[1]
            context = history[:, -window_frames:].detach()
            chunk_initial = None
            generated_frames = image_or_video_shape[1]

        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        noise_shape = list(image_or_video_shape)
        noise_shape[1] = generated_frames
        output, denoised_from, denoised_to, sim_step = (
            self.inference_pipeline.inference_with_trajectory(
                noise=torch.randn(
                    noise_shape, device=self.device, dtype=self.dtype
                ),
                initial_latent=chunk_initial,
                context_latents=context,
                exit_step=state["exit_step"],
                return_sim_step=True,
                **{
                    key: value
                    for key, value in conditional_dict.items()
                    if key != "initial_latent"
                },
            )
        )
        state["exit_step"] = sim_step - 1
        generated = output[:, -generated_frames:].to(self.dtype)
        state["history"] = (
            torch.cat([initial_latent, generated.detach()], dim=1)
            if history is None
            else torch.cat([history, generated.detach()], dim=1)
        )
        return generated, prefix, prefix_start, None, denoised_from, denoised_to

    def _generate_for_loss(
        self,
        image_or_video_shape,
        conditional_dict,
        initial_latent,
        chunk_index,
        phase,
    ):
        if chunk_index is not None:
            return self._run_rollout_generator(
                image_or_video_shape,
                conditional_dict,
                initial_latent,
                chunk_index,
                phase,
            )
        pred, mask, denoised_from, denoised_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            return_full_rollout=True,
        )
        pred, prefix, mask = self._split_context_matched_rollout(pred, mask)
        return pred, prefix, 1, mask, denoised_from, denoised_to

    def _rollout_anchor_loss(self, generated, reference):
        dims = (1, 3, 4)
        generated = generated.float()
        reference = reference[:, :generated.shape[1]].float()
        mean_loss = F.l1_loss(
            generated.mean(dims), reference.mean(dims)
        )
        std_loss = F.l1_loss(
            generated.std(dims, unbiased=False),
            reference.std(dims, unbiased=False),
        )
        return (
            self.rollout_mean_weight * mean_loss
            + self.rollout_std_weight * std_loss
        )

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        rollout_chunk_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        (
            pred_image,
            prefix,
            prefix_start,
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
        ) = self._generate_for_loss(
            image_or_video_shape,
            conditional_dict,
            initial_latent,
            rollout_chunk_index,
            "generator",
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            prefix=prefix,
            prefix_start=prefix_start,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )
        if rollout_chunk_index is not None and (
            self.rollout_mean_weight or self.rollout_std_weight
        ):
            anchor_input = (
                torch.cat([initial_latent, pred_image], dim=1)
                if rollout_chunk_index == 0 else pred_image
            )
            dmd_loss = dmd_loss + self._rollout_anchor_loss(
                anchor_input, clean_latent
            )

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        rollout_chunk_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            (
                generated_image,
                prefix,
                prefix_start,
                _,
                denoised_timestep_from,
                denoised_timestep_to,
            ) = self._generate_for_loss(
                image_or_video_shape,
                conditional_dict,
                initial_latent,
                rollout_chunk_index,
                "critic",
            )

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            generated_image.shape[0],
            generated_image.shape[1],
            self.num_frame_per_block,
            uniform_timestep=True,
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, generated_image.shape[:2])

        noised_history, history_timestep = self._noise_prefix(
            prefix,
            prefix_start,
        )
        pred_fake_flow, pred_fake_image = self._score_with_prefix(
            self.fake_score,
            noisy_generated_image,
            critic_timestep,
            conditional_dict,
            noised_history,
            history_timestep,
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            if self.context_matched_distillation:
                flow_pred = pred_fake_flow.flatten(0, 1)
            else:
                flow_pred = self.generator._convert_x0_to_flow_pred(
                    scheduler=self.scheduler,
                    x0_pred=pred_fake_image.flatten(0, 1),
                    xt=noisy_generated_image.flatten(0, 1),
                    timestep=critic_timestep.flatten(0, 1)
                )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, generated_image.shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
