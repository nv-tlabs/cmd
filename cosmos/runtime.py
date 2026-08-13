"""Small runtime helpers needed by the vendored Cosmos-Predict2.5 DiT.

The training integration does not use NVIDIA's context-parallel or neighborhood
attention paths.  Keeping these helpers local avoids importing the full
``cosmos_predict2`` package (whose top-level import requires a CUDA extra).
"""

import logging
from enum import Enum
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn


log = logging.getLogger(__name__)


class DataType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    MIX = "mix"

    def __str__(self) -> str:
        return self.value


class RMSNorm(nn.Module):
    """RMSNorm with the same single ``weight`` parameter as TE RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight


def apply_rotary_pos_emb(
    value: torch.Tensor,
    freqs: torch.Tensor,
    *,
    tensor_format: str = "bshd",
    fused: bool = True,
) -> torch.Tensor:
    """Apply the split-half RoPE convention used by Predict2.5."""
    del fused
    if tensor_format != "bshd":
        raise ValueError(f"Unsupported rotary tensor format: {tensor_format}")
    if freqs.ndim == 4 and freqs.shape[0] == value.shape[1]:
        freqs = freqs.permute(1, 0, 2, 3)
    first, second = value.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return value * freqs.cos() + rotated * freqs.sin()


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool = False,
    **_kwargs,
) -> torch.Tensor:
    """Flash-SDPA attention for Cosmos tensors shaped ``[B, S, H, D]``."""
    output = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=is_causal,
    )
    return output.transpose(1, 2)


class DotProductAttention(nn.Module):
    """Local replacement for the unused TE attention backend."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return attention(query, key, value, **kwargs).flatten(-2)

    def set_context_parallel_group(self, *args, **kwargs) -> None:
        del args, kwargs


def split_inputs_cp(
    x: torch.Tensor,
    seq_dim: int,
    cp_group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """Split a tensor for the optional context-parallel model path."""
    if cp_group is None or dist.get_world_size(cp_group) == 1:
        return x
    world_size = dist.get_world_size(cp_group)
    if x.shape[seq_dim] % world_size:
        raise ValueError("Context-parallel sequence length must divide world size")
    rank = dist.get_rank(cp_group)
    return x.chunk(world_size, dim=seq_dim)[rank].contiguous()


class MinimalA2AAttnOp(nn.Module):
    """Non-context-parallel equivalent of Predict2.5's minimal A2A op."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return attention(query, key, value, **kwargs).flatten(-2)

    def set_context_parallel_group(self, process_group, *args, **kwargs) -> None:
        del args, kwargs
        if process_group is not None and dist.get_world_size(process_group) > 1:
            raise NotImplementedError(
                "The local Cosmos integration does not enable context parallelism"
            )


class NeighborhoodAttention(nn.Module):
    """Marker for the unsupported sparse-attention option."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        super().__init__()
        raise NotImplementedError(
            "Neighborhood attention is not used by the Cosmos 2B configuration"
        )


class NattenA2AAttnOp(NeighborhoodAttention):
    pass
