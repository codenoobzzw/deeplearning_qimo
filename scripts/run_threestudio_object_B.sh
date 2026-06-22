#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
CONFIG="${CONFIG:-configs/dreamfusion-sd.yaml}"
PROMPT="${PROMPT:-a small yellow rubber duck toy, smooth plastic surface, single object, full body, centered, clean 3D asset, no text, no watermark}"

if [ ! -f third_party/threestudio/launch.py ]; then
  echo "Missing third_party/threestudio. Run scripts/setup_third_party.sh and install the threestudio environment first." >&2
  exit 1
fi

if [ ! -f "third_party/threestudio/$CONFIG" ]; then
  echo "Missing threestudio SDS config: third_party/threestudio/$CONFIG. Inspect third_party/threestudio/configs and choose a DreamFusion/Stable Diffusion SDS config." >&2
  exit 1
fi

bash scripts/run_and_log.sh object_B_threestudio_sds outputs/task1/object_B_text3d \
  bash -lc "cd third_party/threestudio && $PYTHON_BIN launch.py --config $CONFIG --train --gpu $GPU system.prompt_processor.prompt=\"$PROMPT\" system.background.random_aug=true seed=42"
