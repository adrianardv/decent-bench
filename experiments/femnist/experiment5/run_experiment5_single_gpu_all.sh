#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
RUN_LABEL="${RUN_LABEL:-}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="experiments/femnist/checkpoints/experiment5/batch_logs/single_gpu_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

conditions=(
  clean_baseline
  activation_uniform_low
  activation_uniform_high
  activation_markov_high_availability
  activation_markov_low_availability
  compression_topk_low
  compression_topk_high
  compression_qsgd_low
  compression_qsgd_high
  drops_uniform_low
  drops_uniform_high
  combined_uniform_topk_drops
)

run_condition() {
  local condition="$1"
  local label_args=()

  if [[ -n "${RUN_LABEL}" ]]; then
    label_args=(--run-label "${RUN_LABEL}")
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU ${GPU_ID}: starting ${condition}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
    experiments/femnist/experiment5/experiment5_communication_impairments.py \
    --conditions "${condition}" \
    "${label_args[@]}" \
    2>&1 | tee "${LOG_DIR}/${condition}.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU ${GPU_ID}: finished ${condition}"
}

echo "Writing single-GPU batch logs to ${LOG_DIR}"
echo "Using Python: ${PYTHON_BIN}"
echo "Using GPU: ${GPU_ID}"

for condition in "${conditions[@]}"; do
  run_condition "${condition}"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Experiment 5 single-GPU launch complete."
