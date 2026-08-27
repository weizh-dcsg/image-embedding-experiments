#!/usr/bin/env bash
# Wrapper to run test_databricks_connection.py from any working directory.
# Usage: ./run.sh [row_limit]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON_BIN:-python3}"

exec "$PYTHON" "$SCRIPT_DIR/test_databricks_connection.py" "$@"
