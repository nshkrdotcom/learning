#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPORT_DIR="reports/experiments/E002_multitrack_qkv_shift_register"
STANDARD_RUN="runs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1"
STATIC_RUN="runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1"
TRAIN_ROTATION_RUN="runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1"
POSITION_ROTATION_RUN="runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1"

require_summary() {
  local run_dir="$1"
  if [[ ! -f "${run_dir}/evals/run_summary.json" ]]; then
    echo "Missing run summary: ${run_dir}/evals/run_summary.json" >&2
    echo "Run and summarize the corresponding full run before comparing." >&2
    exit 1
  fi
}

require_summary "${STANDARD_RUN}"
require_summary "${STATIC_RUN}"
require_summary "${TRAIN_ROTATION_RUN}"
require_summary "${POSITION_ROTATION_RUN}"

uv run scripts/compare_runs.py \
  --experiment E002_multitrack_qkv_shift_register \
  --baseline "${STANDARD_RUN}" \
  --candidate "${STATIC_RUN}" \
  --json-out "${REPORT_DIR}/comparison_static_global_vs_standard_refactor.json"

uv run scripts/compare_runs.py \
  --experiment E002_multitrack_qkv_shift_register \
  --baseline "${STATIC_RUN}" \
  --candidate "${TRAIN_ROTATION_RUN}" \
  --json-out "${REPORT_DIR}/comparison_train_rotation_global_vs_static_global.json"

uv run scripts/compare_runs.py \
  --experiment E002_multitrack_qkv_shift_register \
  --baseline "${STATIC_RUN}" \
  --candidate "${POSITION_ROTATION_RUN}" \
  --json-out "${REPORT_DIR}/comparison_position_rotation_global_vs_static_global.json"
