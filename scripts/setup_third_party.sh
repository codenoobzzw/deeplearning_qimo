#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p third_party logs

clone_if_missing() {
  local name="$1"
  local url="$2"
  local extra="${3:-}"
  if [ -d "third_party/$name/.git" ]; then
    echo "third_party/$name already exists"
    git -C "third_party/$name" rev-parse HEAD || true
    return 0
  fi
  if [ -d "third_party/$name" ]; then
    local tmp="third_party/${name}_clone_tmp"
    rm -rf "$tmp"
    if [ -n "$extra" ]; then
      git clone $extra "$url" "$tmp"
    else
      git clone "$url" "$tmp"
    fi
    cp -a "$tmp/." "third_party/$name/"
    rm -rf "$tmp"
    git -C "third_party/$name" rev-parse HEAD || true
    return 0
  fi
  if [ -n "$extra" ]; then
    git clone $extra "$url" "third_party/$name"
  else
    git clone "$url" "third_party/$name"
  fi
  git -C "third_party/$name" rev-parse HEAD || true
}

clone_if_missing gaussian-splatting https://github.com/graphdeco-inria/gaussian-splatting "--recursive"
clone_if_missing threestudio https://github.com/threestudio-project/threestudio ""
clone_if_missing lerobot https://github.com/huggingface/lerobot ""
clone_if_missing calvin https://github.com/mees/calvin ""
