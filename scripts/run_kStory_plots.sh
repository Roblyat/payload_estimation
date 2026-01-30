#!/usr/bin/env bash
set -euo pipefail

# Optional args:
#   $1 -> timestamp (e.g., 20260130_025021). If omitted, runner picks latest.
#   $2 -> sweep_id (default: 1).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_DIR="${SCRIPT_DIR}/../services/sweep"
SWEEP_SRC_DIR="${SWEEP_DIR}/src"

TS="${1:-}"
SWEEP_ID="${2:-1}"

PYTHONPATH="${SWEEP_DIR}:${SWEEP_SRC_DIR}" \
python3 "${SWEEP_DIR}/backup_runner/run_kStory_plots.py" \
    --sweep_id "${SWEEP_ID}" \
    ${TS:+--ts "${TS}"}
