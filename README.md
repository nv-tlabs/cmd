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
  <a href="https://hmrishavbandy.github.io/cmd-site/">Project Page</a>
</p>

## Install

```bash
conda create -n causal-cosmos python=3.10 -y
conda activate causal-cosmos
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
python setup.py develop
```

[Cosmos-Predict2.5 2B access](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)

## Dataset preparation

[Instructions](DATASET.md)

## Stage 1: Teacher Pretraining

[`configs/cosmos/causal_flow_finetune.yaml`](configs/cosmos/causal_flow_finetune.yaml)

```bash
source scripts/setup_huggingface.sh
source scripts/setup_wandb.sh

torchrun --standalone --nproc-per-node=8 \
  train.py \
  --config_path configs/cosmos/causal_flow_finetune.yaml \
  --logdir "$COSMOS_RUN_DIR" \
  --wandb-save-dir "$WANDB_DIR"
```

## Stage 2: Context-Matched Distillation

[`configs/cosmos/context_matched_distillation.yaml`](configs/cosmos/context_matched_distillation.yaml)

```bash
source scripts/setup_huggingface.sh

export COSMOS_RUN_DIR="logs/dl3dv_context_matched_distillation"
export WANDB_DIR="${COSMOS_RUN_DIR}/wandb"
export WANDB_JOB_TYPE="context-matched-distillation"
export WANDB_TAGS="cosmos-predict2.5,2b,causal,i2v,context-matched-distillation,dl3dv-1k"
source scripts/setup_wandb.sh

torchrun --standalone --nproc-per-node=8 \
  train.py \
  --config_path configs/cosmos/context_matched_distillation.yaml \
  --logdir "${COSMOS_RUN_DIR}" \
  --wandb-save-dir "${WANDB_DIR}"
```

## Citation

```bibtex
@misc{bandyopadhyay2026contextmatched,
  title  = {Context-Matched Distillation: Teacher Causality for Autoregressive Video Distillation},
  author = {Bandyopadhyay, Hmrishav and Ren, Xuanchi and Huang, Zijian and Wu, Jay Zhangjie and Cao, Tianshi and Li, Ruilong and Chu, Bryan and Fidler, Sanja and Song, Yi-Zhe and Wang, Zian},
  year   = {2026},
  url    = {https://hmrishavbandy.github.io/cmd-site/}
}
```

## Acknowledgements

This project builds on
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) and
[NVIDIA Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5).
