# DL3DV Dataset Preparation

Complete the installation and Hugging Face setup in [README.md](README.md),
then accept access to
[DL3DV videos](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-video).

## Download the 1K videos

```bash
source scripts/setup_huggingface.sh
python scripts/download_dl3dv_videos.py --max-workers 16
```

Output: `data/dl3dv_1k_videos/1K/<scene-id>/video.mp4`

## Generate captions

```bash
torchrun --standalone --nproc-per-node=8 \
  scripts/create_dl3dv_captions.py \
  --input-dir data/dl3dv_1k_videos/1K \
  --output data/dl3dv_captions.json \
  --num-frames 93 \
  --caption-frames 16
```

Output: `data/dl3dv_captions.json`

## Encode the t32 camera dataset

```bash
torchrun --standalone --nproc-per-node=8 \
  scripts/prepare_dl3dv_lmdb.py \
  --latent-frames 32 \
  --camera-conditioning
```

Cameras: `data/dl3dv_1k_vipe_dav3_cameras`

For each scene, preprocessing infers the source-video-to-camera sampling map
from that video's frame count and camera trajectory length. Integer decimation
is preserved when exact (for example, 60 fps video to 20 fps cameras), while
other FPS ratios use uniformly mapped camera-timeline positions.

Output: `data/cosmos_i2v_lmdb_t32_camera`

## Encode the t24 dataset

```bash
torchrun --standalone --nproc-per-node=8 \
  scripts/prepare_dl3dv_lmdb.py \
  --latent-frames 24
```

Output: `data/cosmos_i2v_lmdb_t24`

The t24 dataset uses the same 974 camera-matched videos and the same inferred
per-video camera-timeline FPS as t32, but it does not store camera arrays in the
LMDB records.

DL3DV is distributed under its own terms. Review the
[DL3DV license](https://github.com/DL3DV-10K/Dataset/blob/main/License.md)
before using the data.
