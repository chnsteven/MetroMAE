#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$REPO"

STORAGE="${TFB_FORECASTING_DATASET_PATH:-$(python3 -m config.path_config DATASET_PATH)}"
LINK="$(python3 -m config.path_config DATASET_LINK_PATH)"

mkdir -p "$STORAGE"

if [[ -e "$LINK" && ! -L "$LINK" ]]; then
  echo "Moving existing forecasting data to $STORAGE"
  shopt -s dotglob nullglob
  for item in "$LINK"/*; do
    [[ -e "$item" ]] || continue
    mv -n "$item" "$STORAGE"/
  done
  rmdir "$LINK" 2>/dev/null || rm -rf "$LINK"
fi

mkdir -p "$(dirname "$LINK")"
ln -sfn "$STORAGE" "$LINK"
echo "Linked $LINK -> $STORAGE"
