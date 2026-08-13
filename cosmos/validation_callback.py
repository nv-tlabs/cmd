"""Periodic causal-I2V validation for Cosmos training."""

import time

import torch
import wandb

from pipeline.causal_inference import CausalInferencePipeline
from utils.distributed import barrier


class CosmosValidationCallback:
    def __init__(self, config) -> None:
        self.config = config
        self.interval = int(config.validation_interval)
        self.seed = int(getattr(config, "validation_seed", 12345))
        self.fps = int(getattr(config, "validation_fps", 16))

    def should_run(self, step: int) -> bool:
        return self.interval > 0 and step % self.interval == 0

    @staticmethod
    def _to_uint8_video(pixels: torch.Tensor) -> torch.Tensor:
        # VAE output is [B, T, C, H, W] in [-1, 1].
        return (
            ((pixels.float() * 0.5 + 0.5).clamp(0, 1) * 255)
            .round()
            .to(torch.uint8)
            .cpu()
        )

    def run(self, trainer, batch) -> None:
        if trainer.disable_wandb or trainer.config.no_visualize:
            return
        prompts = list(batch["prompts"])
        # Collated dataset shape is [B, ODE steps, latent frames, C, H, W].
        ground_truth = batch["ode_latent"][:, -1].detach().contiguous()
        batch_size = len(prompts)
        if ground_truth.shape[0] != batch_size:
            raise ValueError(
                "Training-batch validation received different prompt and latent "
                f"batch sizes: {batch_size} and {ground_truth.shape[0]}"
            )
        started_at = time.monotonic()
        if trainer.is_main_process:
            print(
                f"Starting causal I2V validation at step {trainer.step}; "
                f"generating the current training batch ({batch_size} sample(s))",
                flush=True,
            )

        was_generator_training = trainer.model.generator.training
        was_text_encoder_training = trainer.model.text_encoder.training
        trainer.model.generator.eval()
        trainer.model.text_encoder.eval()

        latent_frames = int(self.config.image_or_video_shape[1])
        latent_shape = tuple(self.config.image_or_video_shape[2:])
        if tuple(ground_truth.shape[1:]) != (latent_frames, *latent_shape):
            raise ValueError(
                "Validation latent shape mismatch: expected "
                f"{(latent_frames, *latent_shape)}, got {tuple(ground_truth.shape[1:])}"
            )

        pipeline = CausalInferencePipeline(
            self.config,
            device=trainer.device,
            generator=trainer.model.generator,
            text_encoder=trainer.model.text_encoder,
            vae=trainer.model.vae,
        )
        cuda_device = torch.device("cuda", trainer.device)
        # FSDP parameter views created under inference_mode have no autograd
        # metadata and cannot be reused by the following training forward.
        # no_grad avoids building the validation graph while keeping those
        # views compatible with FSDP's post-backward hook registration.
        with torch.random.fork_rng(devices=[trainer.device]), torch.no_grad():
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            initial_latent = ground_truth[:, :1].to(
                device=cuda_device,
                dtype=trainer.dtype,
            )
            noise = torch.randn(
                (batch_size, latent_frames - 1, *latent_shape),
                device=cuda_device,
                dtype=trainer.dtype,
            )
            _, generated_latent = pipeline.inference(
                noise=noise,
                text_prompts=prompts,
                initial_latent=initial_latent,
                return_latents=True,
                decode=False,
            )
            # Streaming inference stores per-frame K/V tensors on every block.
            # Reset immediately so periodic validation does not carry the full
            # rollout cache into the following training step.
            pipeline._initialize_kv_cache(
                batch_size=batch_size,
                dtype=trainer.dtype,
                device=cuda_device,
                max_frames=latent_frames,
            )

        # FSDP generator/text-encoder inference above must run on every rank.
        # VAE decode and W&B upload are rank-zero-only.
        barrier()
        if trainer.is_main_process:
            with torch.inference_mode():
                trainer.model.vae.to(device=cuda_device, dtype=trainer.dtype).eval()
                generated_pixels = trainer.model.vae.decode_to_pixel(
                    generated_latent,
                    use_cache=False,
                )
                generated_video = self._to_uint8_video(generated_pixels)
                del generated_pixels

                ground_truth_gpu = ground_truth.to(
                    device=cuda_device,
                    dtype=trainer.dtype,
                )
                ground_truth_pixels = trainer.model.vae.decode_to_pixel(
                    ground_truth_gpu,
                    use_cache=False,
                )
                ground_truth_video = self._to_uint8_video(ground_truth_pixels)
                del ground_truth_gpu, ground_truth_pixels

            wandb_log = {
                "validation/train_batch_size": batch_size,
                "validation/wall_time_seconds": time.monotonic() - started_at,
            }
            for sample_index, prompt in enumerate(prompts):
                comparison = torch.cat(
                    [ground_truth_video[sample_index], generated_video[sample_index]],
                    dim=-1,
                ).numpy()
                prefix = f"validation/train_batch_{sample_index:02d}"
                wandb_log[f"{prefix}/gt_vs_generated"] = wandb.Video(
                    comparison,
                    fps=self.fps,
                    format="mp4",
                    caption=(
                        "Left: ground truth | Right: generated\n"
                        f"{prompt}"
                    ),
                )
                wandb_log[f"{prefix}/prompt"] = prompt
            wandb.log(wandb_log, step=trainer.step)
            trainer.model.vae.to("cpu")
            del generated_video, ground_truth_video, comparison, wandb_log
            torch.cuda.empty_cache()
            print(f"Finished causal I2V validation at step {trainer.step}", flush=True)

        del generated_latent, pipeline
        torch.cuda.empty_cache()
        trainer.model.generator.train(was_generator_training)
        trainer.model.text_encoder.train(was_text_encoder_training)
        barrier()
