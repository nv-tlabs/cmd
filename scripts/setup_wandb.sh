#!/usr/bin/env bash

# Source this file before torchrun: source scripts/setup_wandb.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead of executing it: source scripts/setup_wandb.sh" >&2
  exit 2
fi

_wandb_key_file="${WANDB_API_KEY_FILE:-${HOME}/.wandb_key}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  if [[ ! -r "${_wandb_key_file}" ]]; then
    echo "Set WANDB_API_KEY or point WANDB_API_KEY_FILE to a readable key file." >&2
    return 1
  fi
  IFS= read -r WANDB_API_KEY < "${_wandb_key_file}"
  WANDB_API_KEY="${WANDB_API_KEY#WANDB_API_KEY=}"
  WANDB_API_KEY="${WANDB_API_KEY#WANDB_KEY=}"
  WANDB_API_KEY="${WANDB_API_KEY%\"}"
  WANDB_API_KEY="${WANDB_API_KEY#\"}"
  WANDB_API_KEY="${WANDB_API_KEY%\'}"
  WANDB_API_KEY="${WANDB_API_KEY#\'}"
fi
if [[ ${#WANDB_API_KEY} -lt 20 || "${WANDB_API_KEY}" == *[[:space:]]* ]]; then
  unset WANDB_API_KEY
  echo "The W&B API key has an invalid format." >&2
  return 1
fi
export WANDB_API_KEY

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-causal-cosmos25}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-dl3dv-1k}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-flow-matching}"
export WANDB_TAGS="${WANDB_TAGS:-cosmos-predict2.5,2b,causal,i2v,flow-matching,dl3dv-1k}"
export WANDB_MODE="${WANDB_MODE:-online}"
export COSMOS_RUN_DIR="${COSMOS_RUN_DIR:-logs/dl3dv_causal_flow}"
export WANDB_DIR="${WANDB_DIR:-${COSMOS_RUN_DIR}/wandb}"
mkdir -p "${COSMOS_RUN_DIR}" "${WANDB_DIR}"

echo "W&B configured:"
echo "  project=${WANDB_PROJECT}"
echo "  entity=${WANDB_ENTITY:-account default}"
echo "  group=${WANDB_RUN_GROUP}"
echo "  job_type=${WANDB_JOB_TYPE}"

unset _wandb_key_file
