#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash scripts/run_and_log.sh <label> <expected_output_path_or_dash> <command...>" >&2
  exit 2
fi

LABEL="$1"
EXPECTED_OUTPUT="$2"
shift 2

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

START_ISO="$(date -Is)"
START_EPOCH="$(date +%s)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${LABEL}_${STAMP}.log"
MANIFEST="$PROJECT_ROOT/run_manifest.md"

if git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
else
  GIT_COMMIT="no-git"
fi

printf -v CMD_STRING "%q " "$@"

{
  echo "# $LABEL"
  echo "start: $START_ISO"
  echo "project_root: $PROJECT_ROOT"
  echo "git_commit: $GIT_COMMIT"
  echo "expected_output: $EXPECTED_OUTPUT"
  echo "command: $CMD_STRING"
  echo
  echo "## nvidia-smi before"
  nvidia-smi || true
  echo
  echo "## command output"
} > "$LOG_FILE" 2>&1

if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v "$@" >> "$LOG_FILE" 2>&1
  STATUS=$?
else
  "$@" >> "$LOG_FILE" 2>&1
  STATUS=$?
fi

END_ISO="$(date -Is)"
END_EPOCH="$(date +%s)"
DURATION_SEC=$((END_EPOCH - START_EPOCH))

{
  echo
  echo "## nvidia-smi after"
  nvidia-smi || true
  echo
  echo "status: $STATUS"
  echo "end: $END_ISO"
  echo "duration_sec: $DURATION_SEC"
} >> "$LOG_FILE" 2>&1

{
  echo
  echo "## $LABEL"
  echo "- start: $START_ISO"
  echo "- end: $END_ISO"
  echo "- git commit: $GIT_COMMIT"
  echo "- command: \`$CMD_STRING\`"
  echo "- output path: \`$EXPECTED_OUTPUT\`"
  echo "- log: \`${LOG_FILE#$PROJECT_ROOT/}\`"
  echo "- success: $([ "$STATUS" -eq 0 ] && echo yes || echo no)"
  echo "- duration_sec: $DURATION_SEC"
  echo "- gpu note: see nvidia-smi snapshots in the log"
} >> "$MANIFEST"

exit "$STATUS"
