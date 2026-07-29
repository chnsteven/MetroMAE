#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUTODL_TMP_ROOT="${AUTODL_TMP_ROOT:-/root/autodl-tmp}"
STORAGE="${TFB_FORECASTING_DATASET_PATH:-$AUTODL_TMP_ROOT/TFB/dataset/forecasting}"
LINK="$ROOT/dataset/forecasting"

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
