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

import argparse
import os


def _configure_distributed_compile_cache():
    """Keep lazy CUDA compilation from serializing distributed ranks."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return

    shared_root = os.environ.get(
        "SELF_FORCING_COMPILE_CACHE_ROOT",
        os.path.join("/tmp", f"self_forcing_compile_{os.getuid()}"),
    )
    cache_root = os.path.join(
        shared_root,
        f"rank_{local_rank}",
    )
    # Override generic shared cache variables: sharing one compiler directory
    # across local ranks is the failure mode this setup prevents.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        cache_root,
        "inductor",
    )
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, "triton")
    # The Slurm step assigns one CPU to each rank. More compiler workers only
    # contend for that CPU and amplify rank-to-rank compile skew.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    if local_rank == "0":
        print(f"Using rank-local compile caches under {shared_root}", flush=True)


_configure_distributed_compile_cache()

from omegaconf import OmegaConf
import wandb

from trainer import DiffusionTrainer, GANTrainer, ODETrainer, ScoreDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    family_default = os.path.join(os.path.dirname(args.config_path), "default_config.yaml")
    default_config_path = (
        family_default if os.path.exists(family_default)
        else "configs/default_config.yaml"
    )
    default_config = OmegaConf.load(default_config_path)
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
