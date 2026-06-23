#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A no-space symlink avoids several CUDA/C++ build and import edge cases.
if [[ "$ROOT" == *" "* && ! -e /tmp/hw3proj ]]; then
  ln -s "$ROOT" /tmp/hw3proj
fi

ROOT_FOR_TS="${ROOT_FOR_TS:-/tmp/hw3proj}"
THREESTUDIO_ROOT="${THREESTUDIO_ROOT:-$ROOT_FOR_TS/third_party/threestudio}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_FOR_TS/envs/threestudio310/bin/python}"
IMAGE_PATH="${IMAGE_PATH:-load/images/cup_rgba.png}"
GPU="${GPU:-0}"

cd "$THREESTUDIO_ROOT"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hw3_cache/mpl}" \
  HF_HOME="${HF_HOME:-/tmp/hw3_cache/hf}" \
  TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/hw3_cache/hf}" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  "$PYTHON_BIN" launch.py \
    --config configs/stable-zero123.yaml --train \
    data.image_path="$IMAGE_PATH" \
    data.height="${HEIGHT:-64}" \
    data.width="${WIDTH:-64}" \
    data.resolution_milestones=[] \
    data.random_camera.height="${CAMERA_HEIGHT:-32}" \
    data.random_camera.width="${CAMERA_WIDTH:-32}" \
    data.random_camera.batch_size=1 \
    data.random_camera.eval_height=64 \
    data.random_camera.eval_width=64 \
    data.random_camera.n_val_views=4 \
    data.random_camera.n_test_views=12 \
    trainer.max_steps="${MAX_STEPS:-1}" \
    trainer.val_check_interval=1 \
    checkpoint.every_n_train_steps=1 \
    system.renderer.num_samples_per_ray="${NUM_SAMPLES_PER_RAY:-16}" \
    trainer.precision=32
