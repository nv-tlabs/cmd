# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Small runtime helpers needed by the vendored Cosmos-Predict2.5 DiT.

The training integration does not use NVIDIA's context-parallel or neighborhood
attention paths.  Keeping these helpers local avoids importing the full
``cosmos_predict2`` package (whose top-level import requires a CUDA extra).
"""

import logging
from functools import lru_cache
from enum import Enum
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

try:
    from transformer_engine.pytorch.attention import (
        DotProductAttention as _TransformerEngineAttention,
    )
    try:
        from transformer_engine.pytorch.attention.rope import (
            apply_rotary_pos_emb as _transformer_engine_rope,
        )
    except ImportError:
        from transformer_engine.pytorch.attention import (
            apply_rotary_pos_emb as _transformer_engine_rope,
        )
except ImportError:
    _TransformerEngineAttention = None
    _transformer_engine_rope = None


log = logging.getLogger(__name__)


class DataType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    MIX = "mix"

    def __str__(self) -> str:
        return self.value


class RMSNorm(nn.Module):
    """Checkpoint-compatible RMSNorm backed by PyTorch's fused operator."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            value,
            (value.shape[-1],),
            self.weight,
            self.eps,
        )


def apply_rotary_pos_emb(
    value: torch.Tensor,
    freqs: torch.Tensor,
    *,
    tensor_format: str = "bshd",
    fused: bool = True,
) -> torch.Tensor:
    """Apply rotary embeddings with the fused CUDA implementation when present."""
    if tensor_format != "bshd":
        raise ValueError(f"Unsupported rotary tensor format: {tensor_format}")
    if _transformer_engine_rope is not None and value.is_cuda:
        return _transformer_engine_rope(
            value.contiguous(),
            freqs.contiguous(),
            tensor_format=tensor_format,
            fused=fused,
        )
    if freqs.ndim == 4 and freqs.shape[0] == value.shape[1]:
        freqs = freqs.permute(1, 0, 2, 3)
    first, second = value.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return value * freqs.cos() + rotated * freqs.sin()


@lru_cache(maxsize=None)
def _get_transformer_engine_attention(
    num_heads: int,
    head_dim: int,
) -> nn.Module:
    if _TransformerEngineAttention is None:
        raise RuntimeError("Transformer Engine attention is unavailable")
    module = _TransformerEngineAttention(
        num_heads,
        head_dim,
        num_gqa_groups=num_heads,
        attention_dropout=0.0,
        qkv_format="bshd",
        attn_mask_type="no_mask",
    )
    module.eval()
    return module


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool = False,
    **_kwargs,
) -> torch.Tensor:
    """Fused attention for Cosmos tensors shaped ``[B, S, H, D]``."""
    if (
        _TransformerEngineAttention is not None
        and query.is_cuda
        and not is_causal
    ):
        query = query.contiguous().clone()
        key = key.contiguous().clone()
        value = value.contiguous().clone()
        fused_attention = _get_transformer_engine_attention(
            int(query.shape[2]),
            int(query.shape[3]),
        )
        output = fused_attention(
            query,
            key,
            value,
        )
        if isinstance(output, tuple):
            output = output[0]
        if output.ndim == 3:
            output = output.unflatten(-1, (query.shape[2], query.shape[3]))
        return output

    output = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=is_causal,
    )
    return output.transpose(1, 2)


class DotProductAttention(nn.Module):
    """Parameter-free attention module with the expected Cosmos interface."""

    def __init__(self, num_heads: int, head_dim: int, **kwargs) -> None:
        del kwargs
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if query.shape[2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                "Attention input shape does not match the configured heads: "
                f"{tuple(query.shape)}"
            )
        return attention(query, key, value).flatten(-2)

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
