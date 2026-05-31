#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUN_LABEL="${RUN_LABEL:-}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="experiments/femnist/checkpoints/experiment5/batch_logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

run_condition() {
  local gpu_id="$1"
  local condition="$2"
  local label_args=()

  if [[ -n "${RUN_LABEL}" ]]; then
    label_args=(--run-label "${RUN_LABEL}")
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU ${gpu_id}: starting ${condition}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
    experiments/femnist/experiment5/experiment5_communication_impairments.py \
    --conditions "${condition}" \
    "${label_args[@]}" \
    2>&1 | tee "${LOG_DIR}/gpu${gpu_id}_${condition}.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU ${gpu_id}: finished ${condition}"
}

run_batch() {
  local gpu_id="$1"
  shift
  for condition in "$@"; do
    run_condition "${gpu_id}" "${condition}"
  done
}

echo "Writing batch logs to ${LOG_DIR}"
echo "Using Python: ${PYTHON_BIN}"

run_batch 0 \
  clean_baseline \
  activation_uniform_low \
  activation_uniform_high &
pid0=$!

run_batch 1 \
  activation_markov_high_availability \
  activation_markov_low_availability \
  drops_uniform_low &
pid1=$!

run_batch 2 \
  compression_topk_low \
  compression_topk_high \
  compression_qsgd_low &
pid2=$!

run_batch 3 \
  drops_uniform_high \
  compression_qsgd_high \
  combined_uniform_topk_drops &
pid3=$!

wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Experiment 5 batch launch complete."
