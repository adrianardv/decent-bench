"""Run one drift-aware FedLT hyperparameter version for FEMNIST Experiment 0.

Launch several instances with different ``--version-index`` values to evaluate
the grid in parallel on a multi-GPU machine.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from decent_bench import benchmark
from decent_bench.metrics import metric_library as ml

import experiment0 as exp0
from fedlt_stability_tuning import fedlt_candidate


DEFAULT_SWEEP_NAME = f"run_fedlt_drift_versions_{datetime.now():%Y%m%d_%H%M%S}"

ITERATIONS = 1500
N_TRIALS = 3

ADAM_STEP_SIZE_VALUES = [0.0025, 0.005, 0.0075]
ADAM_NUM_LOCAL_EPOCHS_VALUES = [3, 5]
ADAM_RHO_VALUES = [0.1, 1.0, 10.0]
ADAM_BETA1_VALUES = [0.5, 0.9]
ADAM_BETA2_VALUES = [0.999]
ADAM_EPSILON = 1e-8


def build_candidates() -> list[exp0.Candidate]:
    candidates: list[exp0.Candidate] = []
    for step_size, num_local_epochs, rho, beta1, beta2 in product(
        ADAM_STEP_SIZE_VALUES,
        ADAM_NUM_LOCAL_EPOCHS_VALUES,
        ADAM_RHO_VALUES,
        ADAM_BETA1_VALUES,
        ADAM_BETA2_VALUES,
    ):
        candidates.append(
            fedlt_candidate(
                variant=(
                    f"adam_drift_lr_{exp0.format_value(step_size)}_e{num_local_epochs}"
                    f"_rho_{exp0.format_value(rho)}_b1_{exp0.format_value(beta1)}"
                    f"_b2_{exp0.format_value(beta2)}"
                ),
                search_stage="drift_grid",
                step_size=step_size,
                num_local_epochs=num_local_epochs,
                rho=rho,
                local_solver="adam",
                solver_args={"beta1": beta1, "beta2": beta2, "epsilon": ADAM_EPSILON},
            )
        )
    return exp0.deduplicate_candidates(candidates)


def finite_float(value: Any) -> float | None:
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    return float_value if math.isfinite(float_value) else None


def metric_value(table_frame: Any, metric_name: str, *, statistic: str | None = None) -> tuple[float, float]:
    rows = table_frame[table_frame["metric"] == metric_name]
    if statistic is not None:
        statistic_rows = rows[rows["statistic"] == statistic]
        if not statistic_rows.empty:
            rows = statistic_rows
    if rows.empty:
        raise RuntimeError(f"Metric {metric_name!r} with statistic {statistic!r} was not computed.")
    row = rows.iloc[0]
    return float(row["mean"]), float(row["margin_of_error"])


def run_version(
    candidate: exp0.Candidate,
    train_partitions: list[exp0.Dataset],
    validation_data: exp0.Dataset,
    selected_writer_ids: list[str],
    version_path: Path,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    version_path.mkdir(parents=True, exist_ok=True)

    problem, x0 = exp0.build_problem(
        train_partitions,
        validation_data,
        selected_writer_ids,
        state_snapshot_period=max(1, ITERATIONS // 10),
    )
    algorithm = exp0.build_algorithm(candidate, x0, ITERATIONS)
    result = benchmark.benchmark(
        algorithms=[algorithm],
        benchmark_problem=problem,
        n_trials=N_TRIALS,
        max_processes=1,
        progress_step=max(1, ITERATIONS // 10),
        show_speed=True,
        show_trial=True,
        checkpoint_manager=None,
        log_level=logging.INFO,
    )
    metric_result = benchmark.compute_metrics(
        benchmark_result=result,
        table_metrics=[
            ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
            ml.Loss([np.average]),
            ml.ClientDriftFromServer([min, np.average, max], x_log=False, y_log=False),
        ],
        plot_metrics=[
            ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
            ml.Loss([np.average]),
            ml.ClientDriftFromServer([], x_log=False, y_log=False),
        ],
        log_level=logging.INFO,
    )
    benchmark.display_metrics(
        metrics_result=metric_result,
        save_path=version_path / "results",
        show_plots=False,
        log_level=logging.INFO,
    )

    table_frame, _ = metric_result.to_dataframe()
    if table_frame is None:
        raise RuntimeError("No table metrics were computed.")

    metric_result.agent_metrics = None
    exp0.save_pickle_zst(metric_result, version_path / exp0.metric_result_filename)

    server_accuracy, server_accuracy_ci = metric_value(table_frame, "server accuracy")
    accuracy_avg, accuracy_avg_ci = metric_value(table_frame, "accuracy", statistic="avg")
    validation_loss, validation_loss_ci = metric_value(table_frame, "loss", statistic="avg")
    drift_avg, drift_avg_ci = metric_value(table_frame, "client drift from server", statistic="avg")
    drift_max, drift_max_ci = metric_value(table_frame, "client drift from server", statistic="max")

    return {
        "status": "ok",
        "search_stage": candidate.search_stage,
        "candidate_id": candidate.candidate_id,
        "algorithm_key": candidate.algorithm_key,
        "algorithm_group": candidate.group,
        "algorithm_name": candidate.algorithm_name,
        "variant": candidate.variant,
        "server_accuracy_mean": server_accuracy,
        "server_accuracy_margin_of_error": server_accuracy_ci,
        "accuracy_avg_mean": accuracy_avg,
        "accuracy_avg_margin_of_error": accuracy_avg_ci,
        "validation_loss_mean": validation_loss,
        "validation_loss_margin_of_error": validation_loss_ci,
        "client_drift_avg_mean": drift_avg,
        "client_drift_avg_margin_of_error": drift_avg_ci,
        "client_drift_max_mean": drift_max,
        "client_drift_max_margin_of_error": drift_max_ci,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        **exp0.flatten_hyperparameters(candidate.hyperparameters),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-index", type=int, required=True, help="0-based index into the FedLT drift grid.")
    parser.add_argument(
        "--sweep-name",
        default=DEFAULT_SWEEP_NAME,
        help="Shared sweep folder name. Pass the same value to all parallel versions.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/femnist/checkpoints/experiment0/fedlt"),
        help="FedLT checkpoint root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = build_candidates()
    if args.version_index < 0 or args.version_index >= len(candidates):
        raise ValueError(f"--version-index must be in [0, {len(candidates) - 1}]")

    sweep_path = args.root / args.sweep_name
    version_path = sweep_path / f"version_{args.version_index:03d}"
    candidate = candidates[args.version_index]

    train_partitions, validation_data, selected_writer_ids = exp0.load_data()
    metadata = {
        "sweep_name": args.sweep_name,
        "version_index": args.version_index,
        "n_versions": len(candidates),
        "candidate": exp0.best_payload(candidate, row=None),
        "run_config": {
            "algorithm": "fedlt",
            "iterations": ITERATIONS,
            "state_snapshot_period": max(1, ITERATIONS // 10),
            "n_trials": N_TRIALS,
            "selection_fraction": exp0.selection_fraction,
        },
    }
    version_path.mkdir(parents=True, exist_ok=False)
    (version_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        row = run_version(candidate, train_partitions, validation_data, selected_writer_ids, version_path)
    except Exception as exc:  # noqa: BLE001
        logging.exception("FedLT drift version failed: %s", candidate.candidate_id)
        row = {
            "status": "failed",
            "search_stage": candidate.search_stage,
            "candidate_id": candidate.candidate_id,
            "algorithm_key": candidate.algorithm_key,
            "algorithm_group": candidate.group,
            "algorithm_name": candidate.algorithm_name,
            "variant": candidate.variant,
            "error": repr(exc),
            **exp0.flatten_hyperparameters(candidate.hyperparameters),
        }

    (version_path / "version_result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    exp0.cleanup_cuda()
    print(f"FedLT drift version complete: {version_path}")


if __name__ == "__main__":
    main()
