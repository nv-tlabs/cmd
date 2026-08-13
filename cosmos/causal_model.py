# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Block-causal Cosmos-Predict2.5 model with streaming KV cache."""

import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

from cosmos.kv_cache import (
    AttentionOpWithKVCache,
    KVCacheConfig,
    VideoSeqPos,
)
from cosmos.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos.minimal_v4_dit import VideoSize, i4_attention_op


# FlexAttention represents the block mask sparsely, avoiding a dense
# [T*H*W, T*H*W] mask for full-resolution video tokens. Unlike Wan 1.3B,
# Cosmos has a standard 16-head/128-dim layout and does not need expensive
# max-autotune kernel benchmarking.
flex_attention = torch.compile(
    torch_flex_attention,
    dynamic=False,
)


class CausalCosmosAttention(AttentionOpWithKVCache):
    """Causal full-sequence attention and past-only streaming attention."""

    _block_mask_cache: dict[tuple, BlockMask] = {}

    def __init__(self, local_attn_size: int = -1, sink_size: int = 0) -> None:
        if local_attn_size == 0 or local_attn_size < -1:
            raise ValueError("local_attn_size must be -1 or a positive frame count")
        if sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        super().__init__(i4_attention_op)
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def reset_kv_cache(self, max_cache_size: Optional[int] = None) -> None:
        self.k_cache: dict[int, torch.Tensor] = {}
        self.v_cache: dict[int, torch.Tensor] = {}
        self.max_cache_size = max_cache_size

    def _full_sequence_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        video_size: VideoSize,
    ) -> torch.Tensor:
        if q.shape != k.shape or k.shape != v.shape:
            raise ValueError("Full causal attention requires matching Q/K/V shapes")
        tokens_per_frame = video_size.H * video_size.W
        expected_tokens = video_size.T * tokens_per_frame
        if q.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} video tokens, received {q.shape[1]}"
            )
        block_mask, padded_length = self._block_causal_mask(
            device=q.device,
            num_frames=video_size.T,
            tokens_per_frame=tokens_per_frame,
        )

        if padded_length:
            padding = q.new_zeros(
                q.shape[0], padded_length, q.shape[2], q.shape[3]
            )
            q = torch.cat([q, padding], dim=1)
            k = torch.cat([k, padding], dim=1)
            v = torch.cat([v, padding], dim=1)

        output = flex_attention(
            query=q.transpose(1, 2),
            key=k.transpose(1, 2),
            value=v.transpose(1, 2),
            block_mask=block_mask,
        ).transpose(1, 2)
        if padded_length:
            output = output[:, :-padded_length]
        return output.flatten(2)

    def _block_causal_mask(
        self,
        *,
        device: torch.device,
        num_frames: int,
        tokens_per_frame: int,
    ) -> tuple[BlockMask, int]:
        total_length = num_frames * tokens_per_frame
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        cache_key = (
            str(device),
            num_frames,
            tokens_per_frame,
            self.num_frame_per_block,
            self.independent_first_frame,
            self.local_attn_size,
        )
        if cache_key in self._block_mask_cache:
            return self._block_mask_cache[cache_key], padded_length

        padded_total = total_length + padded_length
        block_ends = torch.zeros(padded_total, device=device, dtype=torch.long)
        block_tokens = self.num_frame_per_block * tokens_per_frame
        block_start = 0
        if self.independent_first_frame:
            block_ends[:tokens_per_frame] = tokens_per_frame
            block_start = tokens_per_frame
        for start in range(block_start, total_length, block_tokens):
            end = min(start + block_tokens, total_length)
            block_ends[start:end] = end

        def attention_mask(_batch, _head, query_index, key_index):
            before_block_end = key_index < block_ends[query_index]
            if self.local_attn_size != -1:
                local_start = (
                    block_ends[query_index]
                    - self.local_attn_size * tokens_per_frame
                )
                before_block_end = before_block_end & (key_index >= local_start)
            # Padded queries attend only to themselves so no row is fully masked.
            return before_block_end | (query_index == key_index)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=padded_total,
            KV_LEN=padded_total,
            _compile=False,
            device=device,
        )
        self._block_mask_cache[cache_key] = block_mask
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "Cached Cosmos block-causal attention mask: "
                f"frames={num_frames}, block_frames={self.num_frame_per_block}, "
                f"tokens_per_frame={tokens_per_frame}",
                flush=True,
            )
        return block_mask, padded_length

    def _history_indices(self, current_idx: int) -> list[int]:
        window = self.local_attn_size
        if window == -1:
            window = self.max_cache_size or current_idx + 1
        recent_start = max(self.sink_size, current_idx - window + 1)
        sink = range(min(self.sink_size, current_idx))
        recent = range(recent_start, current_idx)
        return list(sink) + list(recent)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache_cfg: Optional[KVCacheConfig] = None,
        video_size: Optional[VideoSize] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if kv_cache_cfg is None or not kv_cache_cfg.run_with_kv:
            if video_size is None:
                raise ValueError("video_size is required for causal Cosmos attention")
            return self._full_sequence_attention(q, k, v, video_size)

        current_idx = int(kv_cache_cfg.current_idx)
        if kv_cache_cfg.store_kv:
            self.k_cache[current_idx] = k.detach()
            self.v_cache[current_idx] = v.detach()

        history_indices = self._history_indices(current_idx)
        missing = [index for index in history_indices if index not in self.k_cache]
        if missing:
            raise RuntimeError(f"Cosmos KV cache is missing frames: {missing[:4]}")

        history_k = [self.k_cache[index] for index in history_indices]
        history_v = [self.v_cache[index] for index in history_indices]
        if history_k and history_k[0].shape[0] != k.shape[0]:
            if history_k[0].shape[0] != 1:
                raise ValueError(
                    "Cached Cosmos batch cannot be broadcast to the current batch"
                )
            history_k = [item.expand(k.shape[0], *item.shape[1:]) for item in history_k]
            history_v = [item.expand(v.shape[0], *item.shape[1:]) for item in history_v]
        k = torch.cat(history_k + [k], dim=1) if history_k else k
        v = torch.cat(history_v + [v], dim=1) if history_v else v
        return i4_attention_op(q, k, v)

    def set_context_parallel_group(self, *args, **kwargs) -> None:
        del args, kwargs


class CausalCosmosModel(MinimalV1LVGDiT):
    """Weight-compatible causal variant of the bidirectional Cosmos 2.5 DiT."""

    def __init__(
        self,
        *args,
        local_attn_size: int = -1,
        sink_size: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_frame_per_block = 1
        self._independent_first_frame = False
        self.causal_attention_ops = []
        for block in self.blocks:
            attention_op = CausalCosmosAttention(
                local_attn_size=local_attn_size,
                sink_size=sink_size,
            )
            block.self_attn.attn_op = attention_op
            self.causal_attention_ops.append(attention_op)

    @property
    def num_frame_per_block(self) -> int:
        return self._num_frame_per_block

    @num_frame_per_block.setter
    def num_frame_per_block(self, value: int) -> None:
        if value <= 0:
            raise ValueError("num_frame_per_block must be positive")
        self._num_frame_per_block = value
        for attention_op in getattr(self, "causal_attention_ops", []):
            attention_op.num_frame_per_block = value

    @property
    def independent_first_frame(self) -> bool:
        return self._independent_first_frame

    @independent_first_frame.setter
    def independent_first_frame(self, value: bool) -> None:
        self._independent_first_frame = bool(value)
        for attention_op in getattr(self, "causal_attention_ops", []):
            attention_op.independent_first_frame = bool(value)

    def forward_seq(
        self,
        x_B_C_T_H_W: torch.Tensor,
        video_pos: VideoSeqPos,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        kv_cache_cfg: Optional[KVCacheConfig] = None,
    ) -> torch.Tensor:
        """Run one causal sequence chunk using the same blocks and weights."""
        if condition_video_input_mask_B_C_T_H_W is None:
            raise ValueError("condition_video_input_mask_B_C_T_H_W is required")

        x_B_C_T_H_W = torch.cat(
            [
                x_B_C_T_H_W,
                condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W),
            ],
            dim=1,
        )
        x_B_T_H_W_D, _, _ = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )
        _, token_t, token_h, token_w, _ = x_B_T_H_W_D.shape
        if token_t * token_h * token_w != video_pos.size():
            raise ValueError("Cosmos sequence positions do not match the input tokens")

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        timesteps_B_T = timesteps_B_T * self.timestep_scale
        time_embedding, adaln_lora = self.t_embedder(timesteps_B_T)
        time_embedding = self.t_embedding_norm(time_embedding)

        full_t = int(video_pos.pos_t.max().item()) + 1
        full_h = int(video_pos.pos_h.max().item()) + 1
        full_w = int(video_pos.pos_w.max().item()) + 1
        rope = self.pos_embedder.generate_embeddings(
            torch.Size([1, full_t, full_h, full_w, self.model_channels])
        )
        linear_index = (
            video_pos.pos_t * (full_h * full_w)
            + video_pos.pos_h * full_w
            + video_pos.pos_w
        )
        rope = rope.index_select(0, linear_index.to(device=rope.device))

        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                time_embedding,
                crossattn_emb,
                rope_emb_L_1_1_D=rope,
                adaln_lora_B_T_3D=adaln_lora,
                extra_per_block_pos_emb=None,
                kv_cache_cfg=kv_cache_cfg,
            )

        output = self.final_layer(
            x_B_T_H_W_D,
            time_embedding,
            adaln_lora_B_T_3D=adaln_lora,
        )
        return self.unpatchify(output)
