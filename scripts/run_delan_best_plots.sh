#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_delan_best_plots.sh [ts] [runs_path] [folds_path] [hypers_path]
# If ts is omitted, the runner picks the latest summary_delan_best_runs_*.
# Paths are optional overrides; pass "" to skip an override.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_DIR="${SCRIPT_DIR}/../services/sweep"
SWEEP_SRC_DIR="${SWEEP_DIR}/src"

TS="${1:-}"
RUNS_PATH="${2:-}"
FOLDS_PATH="${3:-}"
HYPERS_PATH="${4:-}"
RUN_TAG_OVERRIDE="${5:-}"

PYTHONPATH="${SWEEP_DIR}:${SWEEP_SRC_DIR}" \
python3 "${SWEEP_DIR}/backup_runner/run_delan_best_plots.py" \
    ${TS:+--ts "${TS}"} \
    ${RUNS_PATH:+--summary_runs "${RUNS_PATH}"} \
    ${FOLDS_PATH:+--summary_folds "${FOLDS_PATH}"} \
    ${HYPERS_PATH:+--summary_hypers "${HYPERS_PATH}"} \
    ${RUN_TAG_OVERRIDE:+--run_tag "${RUN_TAG_OVERRIDE}"}
