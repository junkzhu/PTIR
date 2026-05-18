#!/usr/bin/env bash

set -euo pipefail

CUDA_DEVICES="0"
DATA_ROOT="/mnt/sdb1/zjk/dataset/tensoIR"
OUT_DIR="outputs/tensoir"
CONFIG_NAME="apps/nerf_synthetic_3dgrt.yaml"
SCENES=(lego armadillo hotdog ficus)
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 --cuda_device 0,1,2,3 [options] [-- extra hydra args]

Options:
  --cuda_device DEVICES   Comma-separated GPU ids, e.g. 0,1,2,3.
  --data_root PATH        TensoIR dataset root. Default: $DATA_ROOT
  --out_dir PATH          Output directory. Default: $OUT_DIR
  --config_name NAME      Hydra config name. Default: $CONFIG_NAME
  --scenes "A B C"        Space-separated scene list. Default: ${SCENES[*]}
  -h, --help              Show this help.

Example:
  $0 --cuda_device 0,1,2,3
  $0 --cuda_device 0,1 -- n_iterations=7000
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda_device)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --config_name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --scenes)
            read -r -a SCENES <<< "$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

IFS=',' read -r -a GPU_IDS <<< "$CUDA_DEVICES"
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "No CUDA devices provided."
    exit 1
fi

mkdir -p "$OUT_DIR/logs"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$OUT_DIR/.cache}"

run_scene() {
    local scene="$1"
    local gpu_id="$2"
    local log_file="$OUT_DIR/logs/train_${scene}.log"
    local scene_args=()

    if [[ "$scene" == "hotdog" ]]; then
        scene_args+=("loss.use_normal_prior_regularization=true")
    fi

    echo "[$(date '+%F %T')] Starting scene=$scene on CUDA_VISIBLE_DEVICES=$gpu_id"
    {
        echo "scene=$scene"
        echo "cuda_device=$gpu_id"
        echo "config=$CONFIG_NAME"
        echo "path=$DATA_ROOT/$scene"
        echo "out_dir=$OUT_DIR"
        echo "experiment_name=$scene"
        printf 'scene_args=%q ' "${scene_args[@]}"
        echo
        nvidia-smi || true
        CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
            --config-name "$CONFIG_NAME" \
            "path=$DATA_ROOT/$scene" \
            "out_dir=$OUT_DIR" \
            "experiment_name=$scene" \
            "${scene_args[@]}" \
            "${EXTRA_ARGS[@]}"
    } > "$log_file" 2>&1
    echo "[$(date '+%F %T')] Finished scene=$scene on CUDA_VISIBLE_DEVICES=$gpu_id"
}

active_jobs=0
for idx in "${!SCENES[@]}"; do
    scene="${SCENES[$idx]}"
    gpu_id="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"

    run_scene "$scene" "$gpu_id" &
    active_jobs=$((active_jobs + 1))

    if [[ $active_jobs -ge ${#GPU_IDS[@]} ]]; then
        wait -n
        active_jobs=$((active_jobs - 1))
    fi
done

wait
echo "All TensoIR training jobs finished. Logs: $OUT_DIR/logs"
