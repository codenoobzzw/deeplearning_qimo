#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhiwei/miniconda3/bin/conda run -n base python}"
PYTHONPATH_3DGS="${PYTHONPATH_3DGS:-$ROOT/local_pkgs/3dgs}"
SOURCE_DIR="${SOURCE_DIR:-data/object_A_video}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/task1/object_A_3dgs}"
LOG_PREFIX="${LOG_PREFIX:-object_A_video_3dgs}"
ITERATIONS="${ITERATIONS:-1000}"
RESOLUTION="${RESOLUTION:--r 2}"

if [ ! -f third_party/gaussian-splatting/convert.py ]; then
  echo "Missing third_party/gaussian-splatting. Run scripts/setup_third_party.sh and install the 3DGS environment first." >&2
  exit 1
fi

if [ ! -d "$SOURCE_DIR/input" ] || [ -z "$(find "$SOURCE_DIR/input" -maxdepth 1 -type f | head -n 1)" ]; then
  echo "Missing prepared object A images in $SOURCE_DIR/input. Run scripts/extract_frames.py first." >&2
  exit 1
fi

bash scripts/run_and_log.sh "${LOG_PREFIX}_convert" "$SOURCE_DIR/sparse" \
  bash -lc "cd third_party/gaussian-splatting && PYTHONPATH='$PYTHONPATH_3DGS' $PYTHON_BIN convert.py -s ../../$SOURCE_DIR --resize --no_gpu --magick_executable convert"

bash scripts/run_and_log.sh "${LOG_PREFIX}_train_${ITERATIONS}" "$OUTPUT_DIR" \
  bash -lc "cd third_party/gaussian-splatting && PYTHONPATH='$PYTHONPATH_3DGS' $PYTHON_BIN train.py -s ../../$SOURCE_DIR -m ../../$OUTPUT_DIR --eval $RESOLUTION --iterations $ITERATIONS --save_iterations $ITERATIONS --test_iterations $ITERATIONS --disable_viewer"

bash scripts/run_and_log.sh "${LOG_PREFIX}_render" "$OUTPUT_DIR" \
  bash -lc "cd third_party/gaussian-splatting && PYTHONPATH='$PYTHONPATH_3DGS' $PYTHON_BIN render.py -m ../../$OUTPUT_DIR --iteration $ITERATIONS"

bash scripts/run_and_log.sh "${LOG_PREFIX}_metrics" "$OUTPUT_DIR/results.json" \
  bash -lc "cd third_party/gaussian-splatting && PYTHONPATH='$PYTHONPATH_3DGS' $PYTHON_BIN metrics.py -m ../../$OUTPUT_DIR"
