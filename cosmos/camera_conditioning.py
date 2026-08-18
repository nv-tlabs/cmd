# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Camera-ray conditioning for the Cosmos video latent grid."""

from __future__ import annotations

import torch


CAMERA_FEATURE_DIM = 6


def camera_frame_indices(
    num_pixel_frames: int,
    frame_stride: int = 4,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Select pixel-frame cameras aligned with temporally compressed latents."""
    if num_pixel_frames <= 0:
        raise ValueError("num_pixel_frames must be positive")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if (num_pixel_frames - 1) % frame_stride:
        raise ValueError(
            "Camera sequence length must be 1 + k * frame_stride; got "
            f"{num_pixel_frames} frames and stride {frame_stride}"
        )
    return torch.arange(
        0,
        num_pixel_frames,
        frame_stride,
        device=device,
        dtype=torch.long,
    )


def frame_relative_camera_to_world(
    camera_to_world: torch.Tensor,
    num_frame_per_block: int = 1,
) -> torch.Tensor:
    """Express each generated block relative to the prior block boundary."""
    if camera_to_world.ndim != 4 or camera_to_world.shape[-2:] != (4, 4):
        raise ValueError(
            "camera_to_world must have shape [B, T, 4, 4]; got "
            f"{tuple(camera_to_world.shape)}"
        )
    if camera_to_world.shape[1] == 0:
        raise ValueError("camera_to_world must contain at least one frame")
    if num_frame_per_block <= 0:
        raise ValueError("num_frame_per_block must be positive")

    poses = camera_to_world.to(torch.float32)
    frame_indices = torch.arange(
        poses.shape[1],
        device=poses.device,
        dtype=torch.long,
    )
    # With an independent I2V prefix at frame zero, frames 1..C use frame 0,
    # frames C+1..2C use frame C, and so on. For C=1 this reduces to the
    # original previous-frame-relative convention.
    anchor_indices = torch.div(
        torch.clamp(frame_indices - 1, min=0),
        num_frame_per_block,
        rounding_mode="floor",
    ) * num_frame_per_block
    anchors = poses.index_select(1, anchor_indices)
    relative = torch.linalg.solve(anchors, poses)
    return relative


def _per_frame_intrinsics(
    intrinsics: torch.Tensor,
    frame_indices: torch.Tensor,
    num_pixel_frames: int,
) -> torch.Tensor:
    if intrinsics.ndim == 3 and intrinsics.shape[-2:] == (3, 3):
        return intrinsics[:, None].expand(-1, frame_indices.numel(), -1, -1)
    if intrinsics.ndim == 4 and intrinsics.shape[-2:] == (3, 3):
        if intrinsics.shape[1] == 1:
            return intrinsics.expand(-1, frame_indices.numel(), -1, -1)
        if intrinsics.shape[1] != num_pixel_frames:
            raise ValueError(
                "Per-frame intrinsics must match the pixel camera sequence; got "
                f"{intrinsics.shape[1]} and {num_pixel_frames} frames"
            )
        return intrinsics.index_select(1, frame_indices)
    raise ValueError(
        "intrinsics must have shape [B, 3, 3] or [B, T, 3, 3]; got "
        f"{tuple(intrinsics.shape)}"
    )


def camera_rays(
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Return ray origins and unit directions as ``[B, T, H, W, 6]``."""
    if camera_to_world.ndim != 4 or camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must have shape [B, T, 4, 4]")
    if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B, T, 3, 3]")
    if camera_to_world.shape[:2] != intrinsics.shape[:2]:
        raise ValueError("Camera poses and intrinsics must have matching B and T")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("Camera image dimensions must be positive")

    poses = camera_to_world.to(torch.float32)
    calibration = intrinsics.to(device=poses.device, dtype=torch.float32)
    focal_x = calibration[..., 0, 0]
    focal_y = calibration[..., 1, 1]
    if torch.any(focal_x <= 0) or torch.any(focal_y <= 0):
        raise ValueError("Camera focal lengths must be positive")

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(image_height, device=poses.device, dtype=torch.float32) + 0.5,
        torch.arange(image_width, device=poses.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    pixel_x = pixel_x[None, None]
    pixel_y = pixel_y[None, None]
    direction_x = (
        pixel_x - calibration[..., 0, 2, None, None]
    ) / focal_x[..., None, None]
    direction_y = (
        pixel_y - calibration[..., 1, 2, None, None]
    ) / focal_y[..., None, None]
    camera_direction = torch.stack(
        [direction_x, direction_y, torch.ones_like(direction_x)],
        dim=-1,
    )
    camera_direction = torch.nn.functional.normalize(camera_direction, dim=-1)

    rotation = poses[..., :3, :3]
    ray_direction = torch.einsum(
        "btij,bthwj->bthwi",
        rotation,
        camera_direction,
    )
    ray_origin = poses[..., :3, 3][..., None, None, :].expand_as(ray_direction)
    return torch.cat([ray_origin, ray_direction], dim=-1)


def patchify_camera_rays(
    rays: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """Flatten each spatial ray patch into the camera token channels."""
    if rays.ndim != 5 or rays.shape[-1] != CAMERA_FEATURE_DIM:
        raise ValueError("rays must have shape [B, T, H, W, 6]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    batch, frames, height, width, channels = rays.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Camera image {(height, width)} is not divisible by patch size {patch_size}"
        )

    rays_bcthw = rays.permute(0, 4, 1, 2, 3).contiguous()
    token_h = height // patch_size
    token_w = width // patch_size
    return (
        rays_bcthw.reshape(
            batch,
            channels,
            frames,
            token_h,
            patch_size,
            token_w,
            patch_size,
        )
        .permute(0, 1, 4, 6, 2, 3, 5)
        .reshape(
            batch,
            channels * patch_size * patch_size,
            frames,
            token_h,
            token_w,
        )
        .contiguous()
    )


def build_camera_conditioning(
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    frame_stride: int = 4,
    patch_size: int = 16,
    num_frame_per_block: int = 1,
    expected_latent_frames: int | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build block-relative origin/direction camera tokens from pixel cameras."""
    if camera_to_world.ndim != 4 or camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must have shape [B, T, 4, 4]")
    num_pixel_frames = camera_to_world.shape[1]
    indices = camera_frame_indices(
        num_pixel_frames,
        frame_stride,
        device=camera_to_world.device,
    )
    if expected_latent_frames is not None and indices.numel() != expected_latent_frames:
        raise ValueError(
            f"Camera sequence produces {indices.numel()} latent frames; "
            f"expected {expected_latent_frames}"
        )

    sampled_poses = camera_to_world.index_select(1, indices)
    sampled_intrinsics = _per_frame_intrinsics(
        intrinsics.to(device=camera_to_world.device),
        indices,
        num_pixel_frames,
    )
    relative_poses = frame_relative_camera_to_world(
        sampled_poses,
        num_frame_per_block=num_frame_per_block,
    )
    rays = camera_rays(
        relative_poses,
        sampled_intrinsics,
        image_height,
        image_width,
    )
    conditioning = patchify_camera_rays(rays, patch_size=patch_size)
    if output_dtype is not None:
        conditioning = conditioning.to(dtype=output_dtype)
    return conditioning
