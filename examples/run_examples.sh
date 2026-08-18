#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CONFIG_DIR="${REPO_ROOT}/configs/cosmos"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-22}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
DRY_RUN=0

declare -A CONFIG_NAMES=(
  [chunk1-short]=t24_l21_student_context_distillation.yaml
  [chunk1-long]=t24_l21_rollout_context_distillation.yaml
  [chunk4-short]=self_forcing_dmd.yaml
  [chunk4-long]=self_forcing_dmd.yaml
  [chunk1-camera]=t32_l21_camera_student_distillation.yaml
  [chunk4-camera]=self_forcing_dmd.yaml
)
declare -A CHECKPOINT_NAMES=(
  [chunk1-short]=chunk1_short_t24_l21.safetensors
  [chunk1-long]=chunk1_long_t126_l21.safetensors
  [chunk4-short]=chunk4_short_t21_l16.safetensors
  [chunk4-long]=chunk4_long_t121_l16.safetensors
  [chunk1-camera]=chunk1_camera_control_t32_l21.safetensors
  [chunk4-camera]=chunk4_camera_control_t29_l24.safetensors
)
declare -A OUTPUT_FRAMES=(
  [chunk1-short]=24
  [chunk1-long]=126
  [chunk4-short]=21
  [chunk4-long]=121
  [chunk1-camera]=32
  [chunk4-camera]=29
)
declare -A BLOCK_SIZES=(
  [chunk1-short]=1
  [chunk1-long]=1
  [chunk4-short]=4
  [chunk4-long]=4
  [chunk1-camera]=1
  [chunk4-camera]=4
)
declare -A LOCAL_ATTENTION=(
  [chunk1-short]=21
  [chunk1-long]=21
  [chunk4-short]=16
  [chunk4-long]=16
  [chunk1-camera]=21
  [chunk4-camera]=24
)
declare -A CAMERA_TARGETS=(
  [chunk1-camera]=1
  [chunk4-camera]=1
)

usage() {
  cat <<'EOF'
Usage: examples/run_examples.sh [--dry-run] [TARGET]

Targets:
  all           all six runs (default)
  short         chunk1 and chunk4 short
  long          chunk1 and chunk4 long
  chunk1        chunk1 short and long
  chunk4        chunk4 short and long
  chunk1-short
  chunk1-long
  chunk4-short
  chunk4-long
  camera        chunk1 and chunk4 camera-conditioned
  chunk1-camera
  chunk4-camera
EOF
}

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

case "${1:-all}" in
  all) TARGETS=(chunk1-short chunk1-long chunk4-short chunk4-long chunk1-camera chunk4-camera) ;;
  short) TARGETS=(chunk1-short chunk4-short) ;;
  long) TARGETS=(chunk1-long chunk4-long) ;;
  chunk1) TARGETS=(chunk1-short chunk1-long) ;;
  chunk4) TARGETS=(chunk4-short chunk4-long) ;;
  camera) TARGETS=(chunk1-camera chunk4-camera) ;;
  chunk1-short|chunk1-long|chunk4-short|chunk4-long|chunk1-camera|chunk4-camera)
    TARGETS=("$1")
    ;;
  *)
    echo "Unknown target: $1" >&2
    usage >&2
    exit 2
    ;;
esac

for path in \
  "${SCRIPT_DIR}/image.png" \
  "${SCRIPT_DIR}/prompt.txt" \
  "${SCRIPT_DIR}/camera_image.png" \
  "${SCRIPT_DIR}/camera_prompt.txt" \
  "${SCRIPT_DIR}/camera.npz"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing example input: ${path}" >&2
    exit 1
  fi
done

if [[ "${DRY_RUN}" == "0" && "${SKIP_HF_SETUP:-0}" != "1" ]]; then
  # shellcheck source=../scripts/setup_huggingface.sh
  source "${REPO_ROOT}/scripts/setup_huggingface.sh"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

for target in "${TARGETS[@]}"; do
  config_path="${CONFIG_DIR}/${CONFIG_NAMES[${target}]}"
  checkpoint_path="${CHECKPOINT_ROOT}/${CHECKPOINT_NAMES[${target}]}"
  output_dir="${OUTPUT_ROOT}/${target}"
  image_path="${SCRIPT_DIR}/image.png"
  prompt_path="${SCRIPT_DIR}/prompt.txt"
  if [[ "${CAMERA_TARGETS[${target}]:-0}" == "1" ]]; then
    image_path="${SCRIPT_DIR}/camera_image.png"
    prompt_path="${SCRIPT_DIR}/camera_prompt.txt"
  fi
  for path in "${config_path}" "${checkpoint_path}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing run input: ${path}" >&2
      exit 1
    fi
  done

  command=(
    "${PYTHON_BIN}" "${REPO_ROOT}/inference.py"
    --config_path "${config_path}"
    --checkpoint_path "${checkpoint_path}"
    --image_path "${image_path}"
    --prompt_path "${prompt_path}"
    --output_folder "${output_dir}"
    --i2v
    --num_output_frames "${OUTPUT_FRAMES[${target}]}"
    --num_frame_per_block "${BLOCK_SIZES[${target}]}"
    --local_attn_size "${LOCAL_ATTENTION[${target}]}"
    --seed "${SEED}"
    --num_samples "${NUM_SAMPLES}"
    --save_with_index
  )
  if [[ "${CAMERA_TARGETS[${target}]:-0}" == "1" ]]; then
    command+=(
      --camera_path "${SCRIPT_DIR}/camera.npz"
      --camera_conditioning
    )
  fi
  printf '[%s]' "${target}"
  printf ' %q' "${command[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "0" ]]; then
    mkdir -p "${output_dir}"
    "${command[@]}"
  fi
done
