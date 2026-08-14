# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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

from easydict import EasyDict

from .shared_config import wan_shared_cfg

# ------------------------ Wan T2V 14B ------------------------#

t2v_14B = EasyDict(__name__='Config: Wan T2V 14B')
t2v_14B.update(wan_shared_cfg)

# t5
t2v_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
t2v_14B.t5_tokenizer = 'google/umt5-xxl'

# vae
t2v_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
t2v_14B.vae_stride = (4, 8, 8)

# transformer
t2v_14B.patch_size = (1, 2, 2)
t2v_14B.dim = 5120
t2v_14B.ffn_dim = 13824
t2v_14B.freq_dim = 256
t2v_14B.num_heads = 40
t2v_14B.num_layers = 40
t2v_14B.window_size = (-1, -1)
t2v_14B.qk_norm = True
t2v_14B.cross_attn_norm = True
t2v_14B.eps = 1e-6
