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
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TINY_SD_MODEL="${TINY_SD_MODEL:-hf-internal-testing/tiny-stable-diffusion-pipe}"
PROMPT="${PROMPT:-a small yellow rubber duck toy}"
GPU="${GPU:-0}"

cd "$THREESTUDIO_ROOT"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  HF_ENDPOINT="$HF_ENDPOINT" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hw3_cache/mpl}" \
  HF_HOME="${HF_HOME:-/tmp/hw3_cache/hf}" \
  TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/hw3_cache/hf}" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  "$PYTHON_BIN" launch.py \
    --config configs/dreamfusion-sd.yaml --train \
    system.prompt_processor.pretrained_model_name_or_path="$TINY_SD_MODEL" \
    system.guidance.pretrained_model_name_or_path="$TINY_SD_MODEL" \
    system.prompt_processor.prompt="$PROMPT" \
    trainer.max_steps="${MAX_STEPS:-1}" \
    trainer.val_check_interval=1 \
    checkpoint.every_n_train_steps=1 \
    data.width="${WIDTH:-32}" \
    data.height="${HEIGHT:-32}" \
    system.renderer.num_samples_per_ray="${NUM_SAMPLES_PER_RAY:-16}" \
    trainer.precision=32
