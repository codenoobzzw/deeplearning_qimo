#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY_PG="${PY_PG:-/home/zhangzhiwei/miniconda3/bin/conda run -n PG python}"

$PY_PG scripts/prepare_calvin_splits.py --fallback-known-splits
bash scripts/train_act_A.sh
bash scripts/train_act_ABC.sh
bash scripts/eval_act_D.sh
