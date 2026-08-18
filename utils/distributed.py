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

from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def fsdp_state_dict(model):
    if getattr(model, "_self_forcing_uses_fsdp2", False):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        return get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )

    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def create_fsdp2_device_mesh():
    """Create the one-dimensional world mesh used by Cosmos FSDP2 models."""
    if not dist.is_initialized():
        raise RuntimeError("FSDP2 requires an initialized process group")

    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh(
        "cuda",
        (dist.get_world_size(),),
        mesh_dim_names=("fsdp",),
    )


def fsdp2_wrap_cosmos_model(module, mesh, mixed_precision=False):
    """Shard a Cosmos diffusion wrapper at transformer boundaries."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    if not hasattr(module, "model") or not hasattr(module.model, "blocks"):
        raise TypeError("Expected a Cosmos diffusion wrapper with model.blocks")

    device = torch.device("cuda", torch.cuda.current_device())
    module.to(device=device)

    if mixed_precision:
        child_mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        root_mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
    else:
        child_mp_policy = MixedPrecisionPolicy()
        root_mp_policy = MixedPrecisionPolicy()

    for block in module.model.blocks:
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=child_mp_policy,
        )

    # These modules own parameters outside the transformer block stack and are
    # called through their regular Module.forward entry points.
    child_names = ["final_layer", "t_embedder", "x_embedder"]
    if getattr(module.model, "extra_per_block_abs_pos_emb", False):
        child_names.append("extra_pos_embedder")
    if getattr(module.model, "extra_image_context_dim", None) is not None:
        child_names.append("img_context_proj")
    for child_name in child_names:
        child = getattr(module.model, child_name, None)
        if child is not None:
            fully_shard(
                child,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=child_mp_policy,
            )

    fully_shard(
        module,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=root_mp_policy,
    )
    module._self_forcing_uses_fsdp2 = True
    module._self_forcing_fsdp2_mesh = mesh
    return module


@torch.no_grad()
def distributed_clip_grad_norm_(module, max_norm, norm_type=2.0):
    """Clip gradients for either legacy FSDP or a Cosmos FSDP2 wrapper."""
    if not getattr(module, "_self_forcing_uses_fsdp2", False):
        return module.clip_grad_norm_(max_norm, norm_type=norm_type)
    if float(norm_type) != 2.0:
        raise NotImplementedError("The FSDP2 path currently supports L2 clipping only")

    device = torch.device("cuda", torch.cuda.current_device())
    local_squared_norm = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    local_gradients = []
    rank = dist.get_rank()
    for parameter in module.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue

        if hasattr(gradient, "to_local") and hasattr(gradient, "placements"):
            local_gradient = gradient.to_local()
            placements = gradient.placements
            is_replicated = all(
                placement.__class__.__name__ == "Replicate"
                for placement in placements
            )
        else:
            local_gradient = gradient
            is_replicated = True

        local_gradients.append(local_gradient)
        if not is_replicated or rank == 0:
            local_squared_norm.add_(
                local_gradient.detach().float().square().sum()
            )

    mesh = module._self_forcing_fsdp2_mesh
    dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM, group=mesh.get_group())
    total_norm = local_squared_norm.sqrt()
    clip_coefficient = torch.clamp(
        torch.as_tensor(max_norm, device=total_norm.device, dtype=total_norm.dtype)
        / (total_norm + 1e-6),
        max=1.0,
    )
    for local_gradient in local_gradients:
        local_gradient.mul_(clip_coefficient.to(dtype=local_gradient.dtype))
    return total_norm


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
