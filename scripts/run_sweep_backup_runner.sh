#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_DIR="${SCRIPT_DIR}/../services/sweep"
SWEEP_SRC_DIR="${SWEEP_DIR}/src"

PYTHONPATH="${SWEEP_DIR}:${SWEEP_SRC_DIR}" \
python3 "${SWEEP_DIR}/backup_runner/run_lstm_best_plots.py" --ts 20260129_184850
