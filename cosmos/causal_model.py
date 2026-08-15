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

"""Block-causal Cosmos-Predict2.5 model with streaming KV cache."""

import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask
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


_SPARSE_BLOCK_SIZE = 128


def _merge_intervals(
    intervals: list[tuple[int, int]],
    limit: int,
) -> tuple[tuple[int, int], ...]:
    """Clip and merge half-open token intervals."""
    clipped = sorted(
        (max(0, start), min(limit, end))
        for start, end in intervals
        if max(0, start) < min(limit, end)
    )
    merged: list[list[int]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def _full_sequence_query_ranges(
    *,
    total_length: int,
    tokens_per_frame: int,
    num_frame_per_block: int,
    independent_first_frame: bool,
    local_attn_size: int,
    sink_size: int,
) -> list[tuple[int, int, tuple[tuple[int, int], ...]]]:
    """Describe the allowed K/V intervals for full causal attention."""
    block_tokens = num_frame_per_block * tokens_per_frame
    prefix_tokens = tokens_per_frame if independent_first_frame else 0
    use_local_attn = local_attn_size > 0
    sink_tokens = max(sink_size, 0) * tokens_per_frame
    window_tokens = max(local_attn_size - sink_size, 0) * tokens_per_frame

    query_ranges = []
    if prefix_tokens:
        prefix_intervals = (
            [(0, prefix_tokens)]
            if not use_local_attn
            else [(max(prefix_tokens - window_tokens, 0), prefix_tokens)]
        )
        query_ranges.append(
            (
                0,
                prefix_tokens,
                _merge_intervals(prefix_intervals, total_length),
            )
        )
    for query_start in range(prefix_tokens, total_length, block_tokens):
        raw_query_end = query_start + block_tokens
        query_end = min(raw_query_end, total_length)
        if use_local_attn:
            recent_start = max(raw_query_end - window_tokens, 0)
            promoted_sink_end = min(
                query_start,
                sink_tokens,
                recent_start,
            )
            recent_start = max(promoted_sink_end, recent_start)
            intervals = [
                (0, promoted_sink_end),
                (recent_start, raw_query_end),
            ]
        else:
            intervals = [(0, raw_query_end)]
        query_ranges.append(
            (
                query_start,
                query_end,
                _merge_intervals(intervals, total_length),
            )
        )
    return query_ranges


def _teacher_forcing_query_ranges(
    *,
    sequence_length: int,
    tokens_per_frame: int,
    num_frame_per_block: int,
    independent_first_frame: bool,
    local_attn_size: int,
    sink_size: int,
) -> list[tuple[int, int, tuple[tuple[int, int], ...]]]:
    """Describe clean-past plus noisy-current teacher-forcing attention."""
    kv_length = 2 * sequence_length
    block_tokens = num_frame_per_block * tokens_per_frame
    prefix_tokens = tokens_per_frame if independent_first_frame else 0
    use_local_attn = local_attn_size > 0
    sink_tokens = max(sink_size, 0) * tokens_per_frame
    window_tokens = max(local_attn_size - sink_size, 0) * tokens_per_frame

    query_ranges = []
    if prefix_tokens:
        query_ranges.append(
            (
                0,
                prefix_tokens,
                _merge_intervals(
                    [(sequence_length, sequence_length + prefix_tokens)],
                    kv_length,
                ),
            )
        )
    for query_start in range(prefix_tokens, sequence_length, block_tokens):
        raw_query_end = query_start + block_tokens
        query_end = min(raw_query_end, sequence_length)
        if use_local_attn:
            recent_start = max(raw_query_end - window_tokens, 0)
            promoted_sink_end = min(
                query_start,
                sink_tokens,
                recent_start,
            )
            recent_start = max(promoted_sink_end, recent_start)
            intervals = [
                (0, promoted_sink_end),
                (recent_start, query_start),
            ]
        else:
            intervals = [(0, query_start)]
        intervals.append(
            (
                sequence_length + query_start,
                sequence_length + raw_query_end,
            )
        )
        query_ranges.append(
            (
                query_start,
                query_end,
                _merge_intervals(intervals, kv_length),
            )
        )
    return query_ranges


def _build_sparse_block_rows(
    *,
    query_ranges: list[
        tuple[int, int, tuple[tuple[int, int], ...]]
    ],
    q_total: int,
    kv_total: int,
    block_size: int = _SPARSE_BLOCK_SIZE,
) -> tuple[list[int], list[list[int]], list[int], list[list[int]]]:
    """Build exact sparse-tile metadata without a dense token mask.

    Each query range has one fixed union of allowed key intervals. Query and
    key tile boundaries need not align with frame boundaries; tiles crossing a
    boundary are emitted as partial and evaluated by ``mask_mod`` in the
    FlexAttention kernel.
    """
    if q_total <= 0 or kv_total <= 0:
        raise ValueError("Sparse attention lengths must be positive")
    if q_total % block_size or kv_total % block_size:
        raise ValueError("Sparse attention lengths must be block aligned")

    previous_end = 0
    for query_start, query_end, intervals in query_ranges:
        if query_start != previous_end or query_end <= query_start:
            raise ValueError("Query ranges must be contiguous and non-empty")
        if query_end > q_total:
            raise ValueError("Query range exceeds padded query length")
        for key_start, key_end in intervals:
            if not (0 <= key_start < key_end <= kv_total):
                raise ValueError("Key interval exceeds padded key length")
        previous_end = query_end

    q_block_count = q_total // block_size
    kv_block_count = kv_total // block_size
    partial_counts: list[int] = []
    partial_rows: list[list[int]] = []
    full_counts: list[int] = []
    full_rows: list[list[int]] = []

    range_index = 0
    for query_block in range(q_block_count):
        query_start = query_block * block_size
        query_end = query_start + block_size
        while (
            range_index < len(query_ranges)
            and query_ranges[range_index][1] <= query_start
        ):
            range_index += 1

        query_segments = []
        candidate_index = range_index
        cursor = query_start
        fully_covered_query = True
        while (
            candidate_index < len(query_ranges)
            and query_ranges[candidate_index][0] < query_end
        ):
            range_start, range_end, intervals = query_ranges[candidate_index]
            segment_start = max(query_start, range_start)
            segment_end = min(query_end, range_end)
            if segment_start > cursor:
                fully_covered_query = False
            if segment_start < segment_end:
                query_segments.append((segment_start, segment_end, intervals))
                cursor = segment_end
            candidate_index += 1
        if cursor < query_end:
            fully_covered_query = False

        partial_indices: list[int] = []
        full_indices: list[int] = []
        for key_block in range(kv_block_count):
            key_start = key_block * block_size
            key_end = key_start + block_size
            any_allowed = False
            fully_allowed = fully_covered_query and bool(query_segments)
            for _segment_start, _segment_end, intervals in query_segments:
                segment_overlaps = any(
                    interval_start < key_end and key_start < interval_end
                    for interval_start, interval_end in intervals
                )
                any_allowed = any_allowed or segment_overlaps
                segment_contains = any(
                    interval_start <= key_start and key_end <= interval_end
                    for interval_start, interval_end in intervals
                )
                fully_allowed = fully_allowed and segment_contains

            if fully_allowed:
                full_indices.append(key_block)
            elif any_allowed:
                partial_indices.append(key_block)

        partial_counts.append(len(partial_indices))
        full_counts.append(len(full_indices))
        partial_rows.append(
            partial_indices + [0] * (kv_block_count - len(partial_indices))
        )
        full_rows.append(
            full_indices + [0] * (kv_block_count - len(full_indices))
        )

    return partial_counts, partial_rows, full_counts, full_rows


def _block_mask_from_intervals(
    *,
    query_ranges: list[
        tuple[int, int, tuple[tuple[int, int], ...]]
    ],
    q_total: int,
    kv_total: int,
    mask_mod,
    device: torch.device,
) -> BlockMask:
    """Create a FlexAttention BlockMask from compact interval metadata."""
    (
        partial_counts,
        partial_rows,
        full_counts,
        full_rows,
    ) = _build_sparse_block_rows(
        query_ranges=query_ranges,
        q_total=q_total,
        kv_total=kv_total,
    )

    def count_tensor(values: list[int]) -> torch.Tensor:
        return torch.tensor(
            values,
            dtype=torch.int32,
            device=device,
        ).view(1, 1, -1)

    def row_tensor(values: list[list[int]]) -> torch.Tensor:
        return torch.tensor(
            values,
            dtype=torch.int32,
            device=device,
        ).view(1, 1, len(values), -1)

    return BlockMask.from_kv_blocks(
        kv_num_blocks=count_tensor(partial_counts),
        kv_indices=row_tensor(partial_rows),
        full_kv_num_blocks=count_tensor(full_counts),
        full_kv_indices=row_tensor(full_rows),
        BLOCK_SIZE=_SPARSE_BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(q_total, kv_total),
    )


class CausalCosmosAttention(AttentionOpWithKVCache):
    """Causal full-sequence attention and past-only streaming attention."""

    _block_mask_cache: dict[tuple, BlockMask] = {}
    _teacher_mask_cache: dict[tuple, BlockMask] = {}

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
            self.sink_size,
        )
        if cache_key in self._block_mask_cache:
            return self._block_mask_cache[cache_key], padded_length

        padded_total = total_length + padded_length
        block_tokens = self.num_frame_per_block * tokens_per_frame
        prefix_tokens = tokens_per_frame if self.independent_first_frame else 0
        use_local_attn = self.local_attn_size > 0
        sink_tokens = max(self.sink_size, 0) * tokens_per_frame
        window_tokens = (
            max(self.local_attn_size - self.sink_size, 0)
            * tokens_per_frame
        )

        query_ranges = _full_sequence_query_ranges(
            total_length=total_length,
            tokens_per_frame=tokens_per_frame,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
        )

        def block_index(position):
            if prefix_tokens == 0:
                return position // block_tokens
            is_prefix = position < prefix_tokens
            generated_block = (position - prefix_tokens) // block_tokens + 1
            return torch.where(
                is_prefix,
                torch.zeros_like(position),
                generated_block,
            )

        def block_bounds(index):
            if prefix_tokens == 0:
                start = index * block_tokens
                return start, start + block_tokens
            is_prefix = index == 0
            generated_block = index - 1
            start = prefix_tokens + generated_block * block_tokens
            end = start + block_tokens
            return (
                torch.where(is_prefix, torch.zeros_like(index), start),
                torch.where(
                    is_prefix,
                    torch.full_like(index, prefix_tokens),
                    end,
                ),
            )

        def attention_mask(_batch, _head, query_index, key_index):
            valid = (query_index < total_length) & (key_index < total_length)
            query_block = block_index(query_index)
            key_block = block_index(key_index)
            query_start, query_end = block_bounds(query_block)
            allowed = key_block <= query_block
            if use_local_attn:
                zero = query_end - query_end
                recent_start = torch.maximum(
                    query_end - window_tokens,
                    zero,
                )
                promoted_sink_end = torch.minimum(
                    torch.minimum(query_start, zero + sink_tokens),
                    recent_start,
                )
                recent_start = torch.maximum(promoted_sink_end, recent_start)
                allowed = allowed & (
                    (key_index < promoted_sink_end)
                    | (
                        (key_index >= recent_start)
                        & (key_index < query_end)
                    )
                )
            return valid & allowed

        block_mask = _block_mask_from_intervals(
            query_ranges=query_ranges,
            q_total=padded_total,
            kv_total=padded_total,
            mask_mod=attention_mask,
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

    def _teacher_forcing_attention(
        self,
        q: torch.Tensor,
        noisy_k: torch.Tensor,
        noisy_v: torch.Tensor,
        teacher_kv: tuple[torch.Tensor, torch.Tensor],
        video_size: VideoSize,
    ) -> torch.Tensor:
        """Attend to clean past blocks and the noisy current block in one call."""
        clean_k, clean_v = teacher_kv
        if q.shape != noisy_k.shape or noisy_k.shape != noisy_v.shape:
            raise ValueError("Teacher forcing requires matching noisy Q/K/V shapes")
        if clean_k.shape != noisy_k.shape or clean_v.shape != noisy_v.shape:
            raise ValueError("Clean and noisy teacher-forcing K/V shapes must match")

        tokens_per_frame = video_size.H * video_size.W
        sequence_length = video_size.T * tokens_per_frame
        if q.shape[1] != sequence_length:
            raise ValueError(
                f"Expected {sequence_length} video tokens, received {q.shape[1]}"
            )
        block_mask, q_padding, kv_padding = self._teacher_forcing_mask(
            device=q.device,
            num_frames=video_size.T,
            tokens_per_frame=tokens_per_frame,
        )

        if q_padding:
            q = torch.cat(
                [
                    q,
                    q.new_zeros(q.shape[0], q_padding, q.shape[2], q.shape[3]),
                ],
                dim=1,
            )
        k = torch.cat([clean_k, noisy_k], dim=1)
        v = torch.cat([clean_v, noisy_v], dim=1)
        if kv_padding:
            padding = k.new_zeros(
                k.shape[0], kv_padding, k.shape[2], k.shape[3]
            )
            k = torch.cat([k, padding], dim=1)
            v = torch.cat([v, padding], dim=1)

        output = flex_attention(
            query=q.transpose(1, 2),
            key=k.transpose(1, 2),
            value=v.transpose(1, 2),
            block_mask=block_mask,
        ).transpose(1, 2)
        if q_padding:
            output = output[:, :-q_padding]
        return output.flatten(2)

    def _teacher_forcing_mask(
        self,
        *,
        device: torch.device,
        num_frames: int,
        tokens_per_frame: int,
    ) -> tuple[BlockMask, int, int]:
        """Build a sparse [noisy Q, clean K + noisy K] block-causal mask."""
        sequence_length = num_frames * tokens_per_frame
        q_total = math.ceil(sequence_length / 128) * 128
        kv_length = 2 * sequence_length
        kv_total = math.ceil(kv_length / 128) * 128
        q_padding = q_total - sequence_length
        kv_padding = kv_total - kv_length
        cache_key = (
            str(device),
            num_frames,
            tokens_per_frame,
            self.num_frame_per_block,
            self.independent_first_frame,
            self.local_attn_size,
            self.sink_size,
        )
        if cache_key in self._teacher_mask_cache:
            return (
                self._teacher_mask_cache[cache_key],
                q_padding,
                kv_padding,
            )

        block_tokens = self.num_frame_per_block * tokens_per_frame
        prefix_tokens = tokens_per_frame if self.independent_first_frame else 0
        use_local_attn = self.local_attn_size > 0
        sink_tokens = max(self.sink_size, 0) * tokens_per_frame
        window_tokens = (
            max(self.local_attn_size - self.sink_size, 0)
            * tokens_per_frame
        )

        query_ranges = _teacher_forcing_query_ranges(
            sequence_length=sequence_length,
            tokens_per_frame=tokens_per_frame,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
        )

        def attention_mask(_batch, _head, query_index, key_index):
            valid_query = query_index < sequence_length
            valid_key = key_index < kv_length
            clean_key = key_index < sequence_length
            key_position = torch.where(
                clean_key,
                key_index,
                key_index - sequence_length,
            )
            query_is_prefix = query_index < prefix_tokens
            key_is_prefix = key_position < prefix_tokens
            query_block = torch.where(
                query_is_prefix,
                query_index - query_index - 1,
                (query_index - prefix_tokens) // block_tokens,
            )
            key_block = torch.where(
                key_is_prefix,
                key_position - key_position - 1,
                (key_position - prefix_tokens) // block_tokens,
            )
            query_start = torch.where(
                query_is_prefix,
                query_index - query_index,
                prefix_tokens + query_block * block_tokens,
            )
            query_end = torch.where(
                query_is_prefix,
                query_index - query_index + prefix_tokens,
                query_start + block_tokens,
            )

            clean_past = clean_key & (~query_is_prefix) & (
                key_is_prefix | (key_block < query_block)
            )
            if use_local_attn:
                zero = query_end - query_end
                recent_start = torch.maximum(
                    query_end - window_tokens,
                    zero,
                )
                promoted_sink_end = torch.minimum(
                    torch.minimum(query_start, zero + sink_tokens),
                    recent_start,
                )
                recent_start = torch.maximum(
                    promoted_sink_end,
                    recent_start,
                )
                clean_past = clean_past & (
                    (key_position < promoted_sink_end)
                    | (
                        (key_position >= recent_start)
                        & (key_position < query_start)
                    )
                )

            noisy_current = (~clean_key) & (key_block == query_block)
            return valid_query & valid_key & (clean_past | noisy_current)

        block_mask = _block_mask_from_intervals(
            query_ranges=query_ranges,
            q_total=q_total,
            kv_total=kv_total,
            mask_mod=attention_mask,
            device=device,
        )
        self._teacher_mask_cache[cache_key] = block_mask
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "Cached Cosmos teacher-forcing attention mask: "
                f"frames={num_frames}, block_frames={self.num_frame_per_block}, "
                f"tokens_per_frame={tokens_per_frame}",
                flush=True,
            )
        return block_mask, q_padding, kv_padding

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
        teacher_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if teacher_kv is not None:
            if kv_cache_cfg is not None and kv_cache_cfg.run_with_kv:
                raise ValueError("Teacher forcing cannot be combined with KV caching")
            if video_size is None:
                raise ValueError("video_size is required for teacher forcing")
            return self._teacher_forcing_attention(
                q,
                k,
                v,
                teacher_kv,
                video_size,
            )
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

    def forward_teacher_forcing(
        self,
        noisy_x_B_C_T_H_W: torch.Tensor,
        clean_x_B_C_T_H_W: torch.Tensor,
        noisy_timesteps_B_T: torch.Tensor,
        clean_timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        camera_condition_B_C_T_H_W: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score a full noisy sequence from clean strict-past block context.

        A no-grad context pass first caches clean K/V for every layer.  A second
        noisy pass uses those K/V so queries attend clean preceding blocks and
        noisy tokens from their current block.
        """
        if condition_video_input_mask_B_C_T_H_W is None:
            raise ValueError("condition_video_input_mask_B_C_T_H_W is required")
        if noisy_x_B_C_T_H_W.shape != clean_x_B_C_T_H_W.shape:
            raise ValueError("Clean and noisy teacher-forcing inputs must match")
        if noisy_timesteps_B_T.shape != clean_timesteps_B_T.shape:
            raise ValueError("Clean and noisy teacher-forcing timesteps must match")
        if noisy_timesteps_B_T.ndim == 1:
            noisy_timesteps_B_T = noisy_timesteps_B_T.unsqueeze(1)
            clean_timesteps_B_T = clean_timesteps_B_T.unsqueeze(1)
        if noisy_timesteps_B_T.shape[1] != noisy_x_B_C_T_H_W.shape[2]:
            raise ValueError("Teacher-forcing timesteps must cover every frame")

        noisy_input = torch.cat(
            [
                noisy_x_B_C_T_H_W,
                condition_video_input_mask_B_C_T_H_W.type_as(
                    noisy_x_B_C_T_H_W
                ),
            ],
            dim=1,
        )
        clean_input = torch.cat(
            [
                clean_x_B_C_T_H_W,
                condition_video_input_mask_B_C_T_H_W.type_as(
                    clean_x_B_C_T_H_W
                ),
            ],
            dim=1,
        )
        noisy_hidden, rope, noisy_extra_pos = self.prepare_embedded_sequence(
            noisy_input,
            fps=fps,
            padding_mask=padding_mask,
        )
        with torch.no_grad():
            clean_hidden, clean_rope, clean_extra_pos = (
                self.prepare_embedded_sequence(
                    clean_input,
                    fps=fps,
                    padding_mask=padding_mask,
                )
            )
        if clean_hidden.shape != noisy_hidden.shape:
            raise ValueError("Embedded clean and noisy video grids must match")
        if rope is None or clean_rope is None:
            raise ValueError("Causal Cosmos teacher forcing requires RoPE")

        camera = None
        if camera_condition_B_C_T_H_W is not None:
            camera = camera_condition_B_C_T_H_W.permute(
                0, 2, 3, 4, 1
            ).contiguous()
            if camera.shape[:4] != noisy_hidden.shape[:4]:
                raise ValueError(
                    "Camera conditioning does not match the teacher-forcing grid: "
                    f"{tuple(camera.shape)} versus {tuple(noisy_hidden.shape)}"
                )

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        noisy_timesteps_B_T = noisy_timesteps_B_T * self.timestep_scale
        clean_timesteps_B_T = clean_timesteps_B_T * self.timestep_scale
        noisy_time, noisy_adaln_lora = self.t_embedder(noisy_timesteps_B_T)
        noisy_time = self.t_embedding_norm(noisy_time)
        with torch.no_grad():
            clean_time, clean_adaln_lora = self.t_embedder(
                clean_timesteps_B_T
            )
            clean_time = self.t_embedding_norm(clean_time)

        clean_kv_by_layer = []
        with torch.no_grad():
            for block in self.blocks:
                clean_hidden, clean_kv = block(
                    clean_hidden,
                    clean_time,
                    crossattn_emb,
                    rope_emb_L_1_1_D=rope,
                    adaln_lora_B_T_3D=clean_adaln_lora,
                    extra_per_block_pos_emb=clean_extra_pos,
                    camera_B_T_H_W_C=camera,
                    return_self_attn_kv=True,
                )
                clean_kv_by_layer.append(clean_kv)
        del clean_hidden

        for block, clean_kv in zip(
            self.blocks,
            clean_kv_by_layer,
            strict=True,
        ):
            noisy_hidden = block(
                noisy_hidden,
                noisy_time,
                crossattn_emb,
                rope_emb_L_1_1_D=rope,
                adaln_lora_B_T_3D=noisy_adaln_lora,
                extra_per_block_pos_emb=noisy_extra_pos,
                camera_B_T_H_W_C=camera,
                teacher_kv=clean_kv,
            )

        output = self.final_layer(
            noisy_hidden,
            noisy_time,
            adaln_lora_B_T_3D=noisy_adaln_lora,
        )
        return self.unpatchify(output)

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
        camera_condition_B_C_T_H_W: Optional[torch.Tensor] = None,
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
        camera_B_T_H_W_C = None
        if camera_condition_B_C_T_H_W is not None:
            camera_B_T_H_W_C = camera_condition_B_C_T_H_W.permute(
                0, 2, 3, 4, 1
            ).contiguous()
            if camera_B_T_H_W_C.shape[:4] != x_B_T_H_W_D.shape[:4]:
                raise ValueError(
                    "Camera conditioning does not match the causal video grid: "
                    f"{tuple(camera_B_T_H_W_C.shape)} versus {tuple(x_B_T_H_W_D.shape)}"
                )

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
                camera_B_T_H_W_C=camera_B_T_H_W_C,
            )

        output = self.final_layer(
            x_B_T_H_W_D,
            time_embedding,
            adaln_lora_B_T_3D=adaln_lora,
        )
        return self.unpatchify(output)
