#!/usr/bin/env bash
# End-to-end run: build test set -> download images -> embed -> evaluate -> report.
# Creates/uses a local .venv on first run.
#
# Usage: ./run_all.sh [--skip-data]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "==> creating venv at $VENV"
  "${PYTHON_BIN:-python3}" -m venv "$VENV"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
fi

cd "$SCRIPT_DIR"

if [[ "${1:-}" != "--skip-data" ]]; then
  echo "==> 1/6 build test set"
  "$PY" 01_build_test_set.py
  echo "==> 2/6 download images"
  "$PY" 02_download_images.py
  echo "==> 3/6 crop to major object"
  "$PY" 02b_crop_objects.py
fi

echo "==> 4/6 embed"
"$PY" 03_embed.py

echo "==> 5/6 evaluate"
"$PY" 04_evaluate.py

echo "==> 6/6 report"
"$PY" 05_report.py

echo "==> latency benchmark"
"$PY" 07_benchmark_latency.py || echo "(latency benchmark skipped)"

echo
echo "Report: $SCRIPT_DIR/results/EVALUATION_REPORT.md"
