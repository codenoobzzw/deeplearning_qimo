#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhiwei/miniconda3/bin/conda run -n base python}"
LEROBOT_PYTHONPATH="${LEROBOT_PYTHONPATH:-$ROOT/local_pkgs/lerobot:$ROOT/third_party/lerobot/src}"

if [ ! -d local_pkgs/lerobot ]; then
  echo "Missing local LeRobot dependencies. If CALVIN simulation is unavailable, implement/use offline Action L1 evaluation and record that in report/tables/task2_eval_D.csv." >&2
  exit 1
fi
if [ ! -d data/calvin/lerobot_env_D ]; then
  echo "Missing data/calvin/lerobot_env_D." >&2
  exit 1
fi

bash scripts/run_and_log.sh act_A_eval_D outputs/task2/eval_D/act_A \
  bash -lc "PYTHONPATH='$LEROBOT_PYTHONPATH' $PYTHON_BIN -m lerobot.scripts.lerobot_eval --policy.path outputs/weights/act_A_best --dataset.repo_id data/calvin/lerobot_env_D --output_dir outputs/task2/eval_D/act_A"

bash scripts/run_and_log.sh act_ABC_eval_D outputs/task2/eval_D/act_ABC \
  bash -lc "PYTHONPATH='$LEROBOT_PYTHONPATH' $PYTHON_BIN -m lerobot.scripts.lerobot_eval --policy.path outputs/weights/act_ABC_best --dataset.repo_id data/calvin/lerobot_env_D --output_dir outputs/task2/eval_D/act_ABC"
