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

    def _noise_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        prefix_timestep = torch.full(
            prefix.shape[:2],
            self.prefix_noise,
            device=prefix.device,
            dtype=torch.float32,
        )
        return self.scheduler.add_noise(
            prefix.flatten(0, 1),
            torch.randn_like(prefix.flatten(0, 1)),
            prefix_timestep.flatten(0, 1),
        ).unflatten(0, prefix.shape[:2])

    def _score_with_prefix(
        self,
        score_model: torch.nn.Module,
        noisy_suffix: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        noised_history: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score each suffix frame from all preceding student frames as K/V."""
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
        suffix_frames = noisy_suffix.shape[1]
        if noised_history is None:
            raise ValueError(
                "Context-Matched Distillation requires noised student history"
            )
        base_prefix_frames = noised_history.shape[1] - suffix_frames + 1
        if base_prefix_frames < 0:
            raise ValueError(
                "Context-Matched Distillation received invalid student history"
            )

        initial_latent = initial_latent.to(
            device=noisy_suffix.device,
            dtype=noisy_suffix.dtype,
        )
        prefix_timestep = torch.full(
            (timestep.shape[0], 1),
            self.prefix_noise,
            device=timestep.device,
            dtype=timestep.dtype,
        )
        kv_cache = score_model.initialize_kv_cache(
            max_frames=1 + noised_history.shape[1] + 1,
            batch_size=noisy_suffix.shape[0],
            dtype=noisy_suffix.dtype,
            device=noisy_suffix.device,
        )
        with torch.no_grad():
            score_model(
                noisy_image_or_video=initial_latent,
                conditional_dict=conditional_dict,
                timestep=torch.zeros_like(prefix_timestep),
                kv_cache=kv_cache,
                current_start=0,
                store_kv=True,
            )
            for frame_index in range(base_prefix_frames):
                score_model(
                    noisy_image_or_video=noised_history[
                        :, frame_index:frame_index + 1
                    ],
                    conditional_dict=conditional_dict,
                    timestep=prefix_timestep,
                    kv_cache=kv_cache,
                    current_start=frame_index + 1,
                    store_kv=True,
                )

        flow_predictions = []
        x0_predictions = []
        for suffix_index in range(suffix_frames):
            current_start = 1 + base_prefix_frames + suffix_index
            flow_pred, x0_pred = score_model(
                noisy_image_or_video=noisy_suffix[
                    :, suffix_index:suffix_index + 1
                ],
                conditional_dict=conditional_dict,
                timestep=timestep[:, suffix_index:suffix_index + 1],
                kv_cache=kv_cache,
                current_start=current_start,
                store_kv=False,
            )
            flow_predictions.append(flow_pred)
            x0_predictions.append(x0_pred)

            if suffix_index + 1 < suffix_frames:
                history_index = base_prefix_frames + suffix_index
                with torch.no_grad():
                    score_model(
                        noisy_image_or_video=noised_history[
                            :, history_index:history_index + 1
                        ],
                        conditional_dict=conditional_dict,
                        timestep=prefix_timestep,
                        kv_cache=kv_cache,
                        current_start=current_start,
                        store_kv=True,
                    )

        return torch.cat(flow_predictions, dim=1), torch.cat(x0_predictions, dim=1)

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        noised_history: Optional[torch.Tensor] = None,
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
        _, pred_fake_image_cond = self._score_with_prefix(
            self.fake_score,
            noisy_image_or_video,
            timestep,
            conditional_dict,
            noised_history,
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self._score_with_prefix(
                self.fake_score,
                noisy_image_or_video,
                timestep,
                unconditional_dict,
                noised_history,
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self._score_with_prefix(
            self.real_score,
            noisy_image_or_video,
            timestep,
            conditional_dict,
            noised_history,
        )

        _, pred_real_image_uncond = self._score_with_prefix(
            self.real_score,
            noisy_image_or_video,
            timestep,
            unconditional_dict,
            noised_history,
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
                uniform_timestep=True
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
            noised_history = (
                self._noise_prefix(
                    torch.cat([prefix, image_or_video[:, :-1].detach()], dim=1)
                )
                if self.context_matched_distillation
                else None
            )

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                noised_history=noised_history,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
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
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            return_full_rollout=self.context_matched_distillation,
        )
        prefix = None
        if self.context_matched_distillation:
            pred_image, prefix, gradient_mask = self._split_context_matched_rollout(
                pred_image,
                gradient_mask,
            )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            prefix=prefix,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
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
            generated_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                return_full_rollout=self.context_matched_distillation,
            )
            prefix = None
            if self.context_matched_distillation:
                generated_image, prefix, _ = self._split_context_matched_rollout(
                    generated_image,
                    gradient_mask,
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
            uniform_timestep=True
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

        noised_history = (
            self._noise_prefix(
                torch.cat([prefix, generated_image[:, :-1]], dim=1)
            )
            if self.context_matched_distillation
            else None
        )
        pred_fake_flow, pred_fake_image = self._score_with_prefix(
            self.fake_score,
            noisy_generated_image,
            critic_timestep,
            conditional_dict,
            noised_history,
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
