# DL3DV Dataset Preparation

Complete the installation and Hugging Face setup in [README.md](README.md),
then accept the access terms for
[DL3DV 480P frames and poses](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P).

## Download DL3DV

The downloader uses DL3DV's native low-resolution release. It downloads RGB
frames together with the COLMAP/Nerfstudio camera poses; it does not download
the 4K videos.

```bash
./scripts/download_dl3dv_1k.sh
```

The default destination is local to this repository:

```text
data/dl3dv_1k_480p/1K/<scene-hash>/
├── images_8/
└── transforms.json
```

To download fewer scenes for a test:

```bash
./scripts/download_dl3dv_1k.sh --num-scenes 10
```

## Caption DL3DV clips

DL3DV does not include text captions. Generate one caption per 93-frame
training clip with `Qwen/Qwen3-VL-8B-Instruct`. Sixteen chronological frames are
sampled from each clip so Qwen can describe both its contents and camera
motion. Captions are constrained to 50–70 words and complete sentences. On one
eight-GPU node:

```bash
torchrun --standalone --nproc-per-node=8 \
  scripts/create_dl3dv_captions.py \
  --input-dir data/dl3dv_1k_480p/1K \
  --output data/dl3dv_captions.json \
  --num-frames 93 \
  --clip-stride 93 \
  --caption-frames 16 \
  --max-new-tokens 128
```

Caption generation is resumable: each caption is committed atomically to the
shared JSON, and rerunning the same command skips completed clips. Validate
coverage after all workers finish:

```bash
python scripts/create_dl3dv_captions.py \
  --input-dir data/dl3dv_1k_480p/1K \
  --output data/dl3dv_captions.json \
  --num-frames 93 \
  --clip-stride 93 \
  --validate-only
```

## Prepare Cosmos latents

The trainer reads precomputed latent LMDB shards, not the downloaded PNGs.
Create one clip first to validate the model access and GPU environment:

```bash
python scripts/prepare_dl3dv_cosmos_lmdb.py \
  --output-dir data/cosmos_i2v_lmdb_smoke \
  --max-scenes 1 \
  --max-clips 1 \
  --captions-json data/dl3dv_captions.json
```

For a full run, launch one preprocessing process per GPU. On one eight-GPU
node:

```bash
torchrun --standalone --nproc-per-node=8 \
  scripts/prepare_dl3dv_cosmos_lmdb.py \
  --input-dir data/dl3dv_1k_480p/1K \
  --output-dir data/cosmos_i2v_lmdb \
  --width 832 \
  --height 480 \
  --num-frames 93 \
  --clip-stride 93 \
  --captions-json data/dl3dv_captions.json
```

The script infers worker count, global worker ID, and local CUDA device from
`torchrun`. Each GPU receives disjoint clips and writes into its own
collision-free worker directory. The trainer discovers both legacy and
distributed LMDB shards recursively.

Preprocessing is resumable. If it is interrupted, rerun the same command with
the same number of workers and add `--resume`. Resume validates existing shard
shapes and input ordering, continues each worker's partially filled final
shard, and skips already encoded clips. Stop the previous preprocessing
process before launching resume; the script holds an exclusive lock per worker
to prevent duplicate copies of a worker from writing the same output
concurrently.

This produces 3,132 non-overlapping posed clips from the currently downloaded
data; downloaded images without a corresponding pose are excluded. Each record
contains:

- a float16 Cosmos latent with shape `[24, 16, 60, 104]`;
- its text prompt;
- 93 aligned camera-to-world matrices;
- the crop-adjusted camera intrinsics;
- source frame indices and scene hash.

Generic prompt fallback is disabled. Preprocessing fails if an exact clip key
is absent from the Qwen caption JSON.

The camera controls are retained in the LMDB for camera-conditioned training.
[`configs/cosmos/causal_flow_camera_finetune.yaml`](configs/cosmos/causal_flow_camera_finetune.yaml)
uses the existing 93 cameras without translation scaling, samples
`0, 4, ..., 92` to align them with the 24 VAE latents, and conditions every
self-attention block with frame-relative ray origins and directions. Launch it
with the same training command shown in the README, replacing the config path.

DL3DV is distributed under its own terms. Review the
[DL3DV license](https://github.com/DL3DV-10K/Dataset/blob/main/License.md)
before using the data.
