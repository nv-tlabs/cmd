#!/usr/bin/env bash

# Download the official DL3DV 0K-1K split at the 480P tier. This release
# contains ordered RGB frames and their camera poses (the camera controls).
#
# The official 480P tier is the closest published tier to Cosmos' 832x480
# training size. Keep the original frame geometry here so its camera
# intrinsics remain valid; crop/resize frames and intrinsics together during
# dataset preprocessing.

set -euo pipefail

readonly DOWNLOAD_URL="https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/scripts/download.py"

output_dir="data/dl3dv_1k_480p"
clean_cache=true
num_scenes=""

usage() {
    echo "Usage: $0 [--num-scenes N] [--output-dir PATH] [--keep-cache]"
    echo
    echo "  --num-scenes N   Download only the first N scenes from the 1K batch."
    echo "                   By default, all scenes in the batch are downloaded."
    echo "  --output-dir PATH  Destination (default: ./data/dl3dv_1k_480p)."
    echo "  --keep-cache       Retain the Hugging Face download cache."
    echo
    echo "Before running, accept access at:"
    echo "  https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P"
    echo "and authenticate with:"
    echo "  python -m huggingface_hub.cli.hf auth login"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-scenes)
            if [[ $# -lt 2 ]]; then
                echo "error: --num-scenes requires a positive integer" >&2
                exit 2
            fi
            if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "error: --num-scenes requires a positive integer" >&2
                exit 2
            fi
            num_scenes="$2"
            shift 2
            ;;
        --output-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --output-dir requires a path" >&2
                exit 2
            fi
            output_dir="$2"
            shift 2
            ;;
        --keep-cache)
            clean_cache=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

python - <<'PY'
import importlib.util
import sys

missing = [
    package
    for package in ("huggingface_hub", "pandas", "tqdm")
    if importlib.util.find_spec(package) is None
]
if missing:
    print(
        "Missing download dependencies: " + ", ".join(missing) +
        ". Install them with: pip install huggingface_hub pandas tqdm",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf -- "$tmp_dir"
}
trap cleanup EXIT

python - "$DOWNLOAD_URL" "$tmp_dir/download.py" <<'PY'
import sys
import urllib.request

urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY

download_args=(
    --odir "$output_dir"
    --subset 1K
    --resolution 480P
    --file_type images+poses
)

if [[ "$clean_cache" == true ]]; then
    download_args+=(--clean_cache)
fi

if [[ -z "$num_scenes" ]]; then
    echo "Downloading the DL3DV 1K 480P batch with camera controls to: $output_dir"
    python "$tmp_dir/download.py" "${download_args[@]}"
else
    echo "Downloading $num_scenes scene(s) from the DL3DV 1K 480P batch with camera controls to: $output_dir"
    python - "$tmp_dir/download.py" "$output_dir" "$num_scenes" "$clean_cache" <<'PY'
import importlib.util
import sys

download_script, output_dir, num_scenes, clean_cache = sys.argv[1:]
num_scenes = int(num_scenes)
clean_cache = clean_cache == "true"

spec = importlib.util.spec_from_file_location("dl3dv_download", download_script)
dl3dv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl3dv)

repo = "DL3DV/DL3DV-ALL-480P"
if not dl3dv.verify_access(repo):
    raise SystemExit(
        "No access to DL3DV/DL3DV-ALL-480P. Accept its Hugging Face "
        "terms and authenticate before running this script."
    )

items = dl3dv.get_download_list(
    subset_opt="1K",
    hash_name="",
    reso_opt="480P",
    file_type="images+poses",
    output_dir=output_dir,
)
if num_scenes > len(items):
    raise SystemExit(
        f"Requested {num_scenes} scenes, but the 1K batch contains only "
        f"{len(items)} downloadable scenes."
    )

if not dl3dv.download(items[:num_scenes], output_dir, clean_cache):
    raise SystemExit("One or more DL3DV scenes failed to download.")
PY
fi
