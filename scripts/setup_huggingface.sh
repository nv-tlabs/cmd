#!/usr/bin/env bash

# Source this file before torchrun: source scripts/setup_huggingface.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead of executing it: source scripts/setup_huggingface.sh" >&2
  exit 2
fi

_cosmos_hf_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
_cosmos_hf_shared_root="$(dirname -- "${_cosmos_hf_repo_root}")"

# Set COSMOS_HF_HOME to override the repository-adjacent shared cache.
export HF_HOME="${COSMOS_HF_HOME:-${HF_HOME:-${_cosmos_hf_shared_root}/.cache/huggingface}}"
mkdir -p "${HF_HOME}"

_cosmos_hf_token_file="${HF_TOKEN_FILE:-}"
if [[ -z "${HF_TOKEN:-}" && -z "${_cosmos_hf_token_file}" ]]; then
  if [[ -r "${HOME}/.cache/huggingface/token" ]]; then
    _cosmos_hf_token_file="${HOME}/.cache/huggingface/token"
  else
    _cosmos_hf_token_file="${HF_HOME}/token"
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -z "${_cosmos_hf_token_file}" || ! -r "${_cosmos_hf_token_file}" ]]; then
    echo "Set HF_TOKEN or point HF_TOKEN_FILE to a readable token file." >&2
    return 1
  fi
  IFS= read -r HF_TOKEN < "${_cosmos_hf_token_file}"
  HF_TOKEN="${HF_TOKEN#HF_TOKEN=}"
  HF_TOKEN="${HF_TOKEN#HUGGING_FACE_HUB_TOKEN=}"
  HF_TOKEN="${HF_TOKEN%\"}"
  HF_TOKEN="${HF_TOKEN#\"}"
  HF_TOKEN="${HF_TOKEN%\'}"
  HF_TOKEN="${HF_TOKEN#\'}"
fi

if [[ ${#HF_TOKEN} -lt 10 || "${HF_TOKEN}" == *[[:space:]]* ]]; then
  unset HF_TOKEN
  echo "The Hugging Face token has an invalid format." >&2
  return 1
fi

export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

echo "Hugging Face configured."

unset _cosmos_hf_repo_root _cosmos_hf_shared_root _cosmos_hf_token_file
