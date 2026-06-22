#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhiwei/miniconda3/bin/conda run -n base python}"
PYTHONPATH_3DGS="${PYTHONPATH_3DGS:-$ROOT/local_pkgs/3dgs}"
ITERATIONS="${ITERATIONS:-3000}"
RESOLUTION="${RESOLUTION:--r 8}"

if [ ! -f third_party/gaussian-splatting/convert.py ]; then
  echo "Missing third_party/gaussian-splatting. Run scripts/setup_third_party.sh and install the 3DGS environment first." >&2
  exit 1
fi

if [ ! -d data/object_A/input ] || [ -z "$(find data/object_A/input -maxdepth 1 -type f | head -n 1)" ]; then
  echo "Missing prepared object A images. Run scripts/extract_frames.py first." >&2
  exit 1
fi

bash scripts/run_and_log.sh object_A_3dgs_convert data/object_A/sparse \
  bash -lc "cd third_party/gaussian-splatting && PYTHONPATH='$PYTHONPATH_3DGS' $PYTHON_BIN convert.py -s ../../data/object_A --resize --no_gpu"

bash scripts/run_and_log.sh object_A_3dgs_train outputs/task1/object_A_3dgs \
  bash -lc "cd third_party/gaussian-splatting && PYTHONPATH='$PYTHONPATH_3DGS' $PYTHON_BIN train.py -s ../../data/object_A -m ../../outputs/task1/object_A_3dgs --eval $RESOLUTION --iterations $ITERATIONS --disable_viewer"

bash scripts/run_and_log.sh object_A_3dgs_render outputs/task1/object_A_3dgs \
  bash -lc "cd third_party/gaussian-splatting && PYTHONPATH='$PYTHONPATH_3DGS' $PYTHON_BIN render.py -m ../../outputs/task1/object_A_3dgs"

bash scripts/run_and_log.sh object_A_3dgs_metrics outputs/task1/object_A_3dgs/results.json \
  bash -lc "cd third_party/gaussian-splatting && PYTHONPATH='$PYTHONPATH_3DGS' $PYTHON_BIN metrics.py -m ../../outputs/task1/object_A_3dgs"
