#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhiwei/miniconda3/bin/conda run -n base python}"
PYTHONPATH_3DGS="${PYTHONPATH_3DGS:-$ROOT/local_pkgs/3dgs}"
SCENE="${SCENE:-counter}"
ITERATIONS="${ITERATIONS:-3000}"
RESOLUTION="${RESOLUTION:--r 8}"
SCENE_DIR="data/mipnerf360/$SCENE"

if [ ! -f third_party/gaussian-splatting/train.py ]; then
  echo "Missing third_party/gaussian-splatting. Run scripts/setup_third_party.sh and install the 3DGS environment first." >&2
  exit 1
fi

if [ ! -d "$SCENE_DIR" ]; then
  echo "Missing Mip-NeRF 360 scene directory: $SCENE_DIR" >&2
  exit 1
fi

bash scripts/run_and_log.sh "background_${SCENE}_3dgs_train" "outputs/task1/background_3dgs/$SCENE" \
  bash -lc "cd third_party/gaussian-splatting && PYTHONPATH='$PYTHONPATH_3DGS' $PYTHON_BIN train.py -s ../../$SCENE_DIR -m ../../outputs/task1/background_3dgs/$SCENE --eval $RESOLUTION --iterations $ITERATIONS --disable_viewer"

bash scripts/run_and_log.sh "background_${SCENE}_3dgs_render" "outputs/task1/background_3dgs/$SCENE" \
  bash -lc "cd third_party/gaussian-splatting && PYTHONPATH='$PYTHONPATH_3DGS' $PYTHON_BIN render.py -m ../../outputs/task1/background_3dgs/$SCENE"

bash scripts/run_and_log.sh "background_${SCENE}_3dgs_metrics" "outputs/task1/background_3dgs/$SCENE/results.json" \
  bash -lc "cd third_party/gaussian-splatting && PYTHONPATH='$PYTHONPATH_3DGS' $PYTHON_BIN metrics.py -m ../../outputs/task1/background_3dgs/$SCENE"
