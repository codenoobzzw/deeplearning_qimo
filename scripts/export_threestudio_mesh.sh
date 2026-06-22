#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/export_threestudio_mesh.sh <trial_dir> [output_label]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
TRIAL_DIR="$1"
LABEL="${2:-threestudio_export}"

if [ ! -f third_party/threestudio/launch.py ]; then
  echo "Missing third_party/threestudio." >&2
  exit 1
fi
if [ ! -f "$TRIAL_DIR/configs/parsed.yaml" ] || [ ! -f "$TRIAL_DIR/ckpts/last.ckpt" ]; then
  echo "Missing parsed.yaml or ckpts/last.ckpt under $TRIAL_DIR" >&2
  exit 1
fi

bash scripts/run_and_log.sh "$LABEL" "$TRIAL_DIR/export" \
  bash -lc "cd third_party/threestudio && $PYTHON_BIN launch.py --config ../../$TRIAL_DIR/configs/parsed.yaml --export --gpu $GPU resume=../../$TRIAL_DIR/ckpts/last.ckpt system.exporter_type=mesh-exporter"
