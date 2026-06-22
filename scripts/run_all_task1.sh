#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY_PG="${PY_PG:-/home/zhangzhiwei/miniconda3/bin/conda run -n PG python}"

$PY_PG scripts/extract_frames.py
$PY_PG scripts/preprocess_object_c.py

bash scripts/train_3dgs_object.sh
bash scripts/run_threestudio_object_B.sh
bash scripts/run_zero123_object_C.sh
bash scripts/train_3dgs_background.sh
$PY_PG scripts/merge_gaussians.py
$PY_PG scripts/render_merged_path.py
