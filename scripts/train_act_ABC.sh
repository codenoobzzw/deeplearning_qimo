#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhiwei/miniconda3/bin/conda run -n base python}"
LEROBOT_PYTHONPATH="${LEROBOT_PYTHONPATH:-$ROOT/local_pkgs/lerobot:$ROOT/third_party/lerobot/src}"

if [ ! -d local_pkgs/lerobot ]; then
  echo "Missing local LeRobot dependencies. Run the install_lerobot_* commands recorded in run_manifest.md or install lerobot[training]." >&2
  exit 1
fi
if [ ! -d data/calvin/lerobot_env_ABC ]; then
  echo "Missing data/calvin/lerobot_env_ABC. Run scripts/prepare_calvin_splits.py after dataset metadata is available." >&2
  exit 1
fi

bash scripts/run_and_log.sh act_ABC_train outputs/task2/act_ABC \
  bash -lc "PYTHONPATH='$LEROBOT_PYTHONPATH' $PYTHON_BIN -m lerobot.scripts.lerobot_train --config_path configs/act_ABC.yaml"
