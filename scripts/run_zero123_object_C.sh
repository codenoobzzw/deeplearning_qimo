#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
CONFIG="${CONFIG:-configs/stable-zero123.yaml}"
PROMPT="${PROMPT:-a black cylindrical thermos bottle with a black lid, a white front handle, white cartoon duck line drawings and white doodle text on the surface, single clean object}"

if [ ! -f third_party/threestudio/launch.py ]; then
  echo "Missing third_party/threestudio. Run scripts/setup_third_party.sh and install the threestudio environment first." >&2
  exit 1
fi
if [ ! -f third_party/threestudio/load/images/cup_rgba.png ]; then
  echo "Missing cup RGBA. Run scripts/preprocess_object_c.py first." >&2
  exit 1
fi
if [ ! -f "third_party/threestudio/$CONFIG" ]; then
  echo "Missing Zero123/Magic123 config: third_party/threestudio/$CONFIG" >&2
  exit 1
fi

bash scripts/run_and_log.sh object_C_zero123 outputs/task1/object_C_image3d \
  bash -lc "cd third_party/threestudio && $PYTHON_BIN launch.py --config $CONFIG --train --gpu $GPU data.image_path=load/images/cup_rgba.png system.prompt_processor.prompt=\"$PROMPT\" data.default_elevation_deg=10 data.default_azimuth_deg=0 seed=42"
