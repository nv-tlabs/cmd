<h1 align="center">Context-Matched Distillation:<br>Teacher Causality for Autoregressive Video Distillation</h1>

<p align="center">
  <strong>Hmrishav Bandyopadhyay<sup>1,2</sup></strong> &nbsp;
  <strong>Xuanchi Ren<sup>1</sup></strong> &nbsp;
  <strong>Zijian Huang<sup>1</sup></strong> &nbsp;
  <strong>Jay Zhangjie Wu<sup>1</sup></strong> &nbsp;
  <strong>Tianshi Cao<sup>1</sup></strong><br>
  <strong>Ruilong Li<sup>1</sup></strong> &nbsp;
  <strong>Bryan Chu<sup>1</sup></strong> &nbsp;
  <strong>Sanja Fidler<sup>1</sup></strong> &nbsp;
  <strong>Yi-Zhe Song<sup>2</sup></strong> &nbsp;
  <strong>Zian Wang<sup>1</sup></strong>
</p>


<p align="center">
  <sup>1</sup>NVIDIA &nbsp;&nbsp; <sup>2</sup>SketchX, CVSSP, University of Surrey<br>
  <a href="https://hmrishavbandy.github.io/cmd-site/">Project Page</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2608.13391">arXiv</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/nvidia/cmd">Hugging Face</a>
</p>



https://github.com/user-attachments/assets/a35265f9-ac64-4fa4-bc87-8166fc2ee338



## Setup

### Installation

```bash
conda create -n causal-cosmos python=3.10 -y
conda activate causal-cosmos
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
python setup.py develop
```

[Cosmos-Predict2.5 2B access](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)

### Checkpoints

```bash
hf download nvidia/cmd --local-dir checkpoints
```

## Dataset preparation

[Dataset preparation instructions](DATASET.md)

The DL3DV setup is provided only to verify that the training pipeline runs end
to end. We do not release the training data used for the reported models.
Reproducing the reported training quality requires a sufficiently large and
diverse dataset; DL3DV alone is not sufficient.

## Inference

The examples use the inputs in `examples/`, load checkpoints from `checkpoints/`,
and write videos to `examples/outputs/`.
The camera-conditioned example is taken from
[SANA-WM-Bench](https://huggingface.co/datasets/Efficient-Large-Model/SANA-WM-Bench).

```bash
# Run all short, long, and camera-conditioned examples.
bash examples/run_examples.sh

# Run one example.
bash examples/run_examples.sh chunk1-short
bash examples/run_examples.sh chunk1-long
bash examples/run_examples.sh chunk4-short
bash examples/run_examples.sh chunk4-long
bash examples/run_examples.sh chunk1-camera
bash examples/run_examples.sh chunk4-camera
```

## Training

### Stage 1: Teacher pretraining

#### Non-camera teacher: T24 / L21

[`configs/cosmos/t24_l21_teacher_causal_flow.yaml`](configs/cosmos/t24_l21_teacher_causal_flow.yaml)

```bash
source scripts/setup_huggingface.sh

export COSMOS_RUN_DIR="logs/t24_l21_teacher"
export WANDB_DIR="${COSMOS_RUN_DIR}/wandb"
source scripts/setup_wandb.sh

torchrun --standalone --nproc-per-node=8 \
  train.py \
  --config_path configs/cosmos/t24_l21_teacher_causal_flow.yaml \
  --logdir "$COSMOS_RUN_DIR" \
  --wandb-save-dir "$WANDB_DIR"
```

#### Camera teacher: T32 / L21

[`configs/cosmos/t32_l21_camera_teacher_causal_flow.yaml`](configs/cosmos/t32_l21_camera_teacher_causal_flow.yaml)

```bash
source scripts/setup_huggingface.sh
export COSMOS_RUN_DIR="logs/t32_l21_camera_teacher"
export WANDB_DIR="${COSMOS_RUN_DIR}/wandb"
source scripts/setup_wandb.sh
torchrun --standalone --nproc-per-node=8 \
  train.py \
  --config_path configs/cosmos/t32_l21_camera_teacher_causal_flow.yaml \
  --logdir "${COSMOS_RUN_DIR}" \
  --wandb-save-dir "${WANDB_DIR}"
```

### Stage 2: Context-matched distillation

#### Non-camera student: T24 / L21

[`configs/cosmos/t24_l21_student_context_distillation.yaml`](configs/cosmos/t24_l21_student_context_distillation.yaml)

```bash
source scripts/setup_huggingface.sh

export COSMOS_RUN_DIR="logs/t24_l21_student_context_distillation"
export WANDB_DIR="${COSMOS_RUN_DIR}/wandb"
export WANDB_JOB_TYPE="context-matched-distillation"
export WANDB_TAGS="cosmos-predict2.5,2b,causal,i2v,context-matched-distillation,dl3dv-1k"
source scripts/setup_wandb.sh

torchrun --standalone --nproc-per-node=8 \
  train.py \
  --config_path configs/cosmos/t24_l21_student_context_distillation.yaml \
  --logdir "${COSMOS_RUN_DIR}" \
  --wandb-save-dir "${WANDB_DIR}"
```

#### Camera student: T32 / L21

[`configs/cosmos/t32_l21_camera_student_distillation.yaml`](configs/cosmos/t32_l21_camera_student_distillation.yaml)

```bash
source scripts/setup_huggingface.sh

export COSMOS_RUN_DIR="logs/t32_l21_camera_student_distillation"
export WANDB_DIR="${COSMOS_RUN_DIR}/wandb"
source scripts/setup_wandb.sh

torchrun --standalone --nproc-per-node=8 \
  train.py \
  --config_path configs/cosmos/t32_l21_camera_student_distillation.yaml \
  --logdir "${COSMOS_RUN_DIR}" \
  --wandb-save-dir "${WANDB_DIR}"
```

### Stage 3: Long-video distillation

[`configs/cosmos/t24_l21_rollout_context_distillation.yaml`](configs/cosmos/t24_l21_rollout_context_distillation.yaml)

```bash
source scripts/setup_huggingface.sh
export COSMOS_RUN_DIR="logs/t24_l21_rollout_context_distillation"
export WANDB_DIR="${COSMOS_RUN_DIR}/wandb"
source scripts/setup_wandb.sh

torchrun --standalone --nproc-per-node=8 \
  train.py \
  --config_path configs/cosmos/t24_l21_rollout_context_distillation.yaml \
  --logdir "${COSMOS_RUN_DIR}" \
  --wandb-save-dir "${WANDB_DIR}"
```

## Citation

```bibtex
@article{bandyopadhyay2026context,
  title   = {Context-Matched Distillation: Teacher Causality for Autoregressive Video Distillation},
  author  = {Bandyopadhyay, Hmrishav and Ren, Xuanchi and Huang, Zijian
             and Wu, Jay Zhangjie and Cao, Tianshi and Li, Ruilong
             and Chu, Bryan and Fidler, Sanja and Song, Yi-Zhe
             and Wang, Zian},
  journal = {arXiv preprint arXiv:2608.13391},
  year    = {2026},
  eprint  = {2608.13391},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url     = {https://arxiv.org/abs/2608.13391}
}
```

## Acknowledgements

This project builds on
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) and
[NVIDIA Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5).
