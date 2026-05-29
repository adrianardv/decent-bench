"""Focused FedLT Adam vs Nesterov tuning for FEMNIST Experiment 0."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np

import experiment0 as exp0


ITERATIONS = 1000
FINAL_ITERATIONS = 2000
N_TRIALS = 1

NESTEROV_STEP_SIZE_VALUES = [0.001, 0.005, 0.01]
NESTEROV_LOCAL_EPOCH_VALUES = [3, 5]
NESTEROV_RHO_VALUES = [0.1, 1.0]
NESTEROV_MOMENTUM_VALUES = [0.3, 0.5, 0.7]

ADAM_STEP_SIZE_VALUES = [0.005, 0.01]
ADAM_LOCAL_EPOCH_VALUES = [5, 8]
ADAM_RHO_VALUES = [0.1, 1.0]
ADAM_BETA1_VALUES = [0.5, 0.9]
ADAM_BETA2_VALUES = [0.99, 0.999]
ADAM_EPSILON_VALUES = [1e-8]


def fedlt_candidate(*, variant, stage, step_size, num_local_epochs, rho, local_solver, solver_args=None):
    hyperparameters = {
        "step_size": step_size,
        "num_local_epochs": num_local_epochs,
        "rho": rho,
        "local_solver": local_solver,
    }
    if solver_args is not None:
        hyperparameters["solver_args"] = solver_args

    return exp0.Candidate(
        algorithm_key="fedlt",
        group="FedLT",
        algorithm_name="FedLT",
        variant=variant,
        search_stage=stage,
        hyperparameters=hyperparameters,
    )


def build_candidates():
    candidates = []

    for step_size in NESTEROV_STEP_SIZE_VALUES:
        for num_local_epochs in NESTEROV_LOCAL_EPOCH_VALUES:
            for rho in NESTEROV_RHO_VALUES:
                for momentum in NESTEROV_MOMENTUM_VALUES:
                    candidates.append(
                        fedlt_candidate(
                            variant=f"nesterov_lr_{step_size:g}_e{num_local_epochs}_rho_{rho:g}_m_{momentum:g}",
                            stage="grid",
                            step_size=step_size,
                            num_local_epochs=num_local_epochs,
                            rho=rho,
                            local_solver="nesterov",
                            solver_args={"momentum": momentum},
                        )
                    )

    for step_size in ADAM_STEP_SIZE_VALUES:
        for num_local_epochs in ADAM_LOCAL_EPOCH_VALUES:
            for rho in ADAM_RHO_VALUES:
                for beta1 in ADAM_BETA1_VALUES:
                    for beta2 in ADAM_BETA2_VALUES:
                        for epsilon in ADAM_EPSILON_VALUES:
                            candidates.append(
                                fedlt_candidate(
                                    variant=(
                                        f"adam_lr_{step_size:g}_e{num_local_epochs}_"
                                        f"rho_{rho:g}_b1_{beta1:g}_b2_{beta2:g}_eps_{epsilon:g}"
                                    ),
                                    stage="grid",
                                    step_size=step_size,
                                    num_local_epochs=num_local_epochs,
                                    rho=rho,
                                    local_solver="adam",
                                    solver_args={
                                        "beta1": beta1,
                                        "beta2": beta2,
                                        "epsilon": epsilon,
                                    },
                                )
                            )

    candidates.append(
        fedlt_candidate(
            variant="reference_adam_lr_0p01_e5_rho_1",
            stage="reference",
            step_size=0.01,
            num_local_epochs=5,
            rho=1.0,
            local_solver="adam",
            solver_args={"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
        )
    )
    candidates.append(
        fedlt_candidate(
            variant="reference_gd_lr_0p0244_e3_rho_1",
            stage="reference",
            step_size=0.02441691061516309,
            num_local_epochs=3,
            rho=1.0,
            local_solver="gd",
        )
    )

    return exp0.deduplicate_candidates(candidates)


def finite_loss(row):
    value = float(row["validation_loss_mean"])
    return value if math.isfinite(value) else float("inf")


def choose_best(rows):
    ok_rows = [row for row in rows if row["status"] == "ok"]
    finite_rows = [row for row in ok_rows if math.isfinite(float(row["validation_loss_mean"]))]
    source_rows = finite_rows or ok_rows
    if not source_rows:
        raise RuntimeError("No successful candidates were available.")
    return max(source_rows, key=lambda row: (float(row["server_accuracy_mean"]), -finite_loss(row)))


def solver_rows(rows, solver):
    return [row for row in rows if row["status"] == "ok" and row.get("local_solver") == solver]


def save_dataset_metadata(config, selected_writer_ids, validation_data, train_partitions, candidate_count):
    payload = {
        "selected_writer_ids": list(selected_writer_ids),
        "n_validation_samples": len(validation_data),
        "n_train_samples_after_validation_split": sum(len(partition) for partition in train_partitions),
        "run_config": {
            "algorithm": "fedlt",
            "iterations": config.iterations,
            "final_iterations": config.final_iterations,
            "n_trials": config.n_trials,
            "run_final": True,
            "selection_fraction": exp0.selection_fraction,
            "candidate_count": candidate_count,
            "nesterov": {
                "step_size_values": NESTEROV_STEP_SIZE_VALUES,
                "num_local_epochs_values": NESTEROV_LOCAL_EPOCH_VALUES,
                "rho_values": NESTEROV_RHO_VALUES,
                "momentum_values": NESTEROV_MOMENTUM_VALUES,
            },
            "adam": {
                "step_size_values": ADAM_STEP_SIZE_VALUES,
                "num_local_epochs_values": ADAM_LOCAL_EPOCH_VALUES,
                "rho_values": ADAM_RHO_VALUES,
                "beta1_values": ADAM_BETA1_VALUES,
                "beta2_values": ADAM_BETA2_VALUES,
                "epsilon_values": ADAM_EPSILON_VALUES,
            },
        },
        "notes": (
            "Focused FedLT Adam vs Nesterov tuning. Candidates are tuned for "
            "1000 iterations; the best Adam and best Nesterov candidates are "
            "then plotted together for 2000 iterations."
        ),
    }
    (config.run_path / "exp0_dataset_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_best_by_solver(config, candidates, rows):
    best_overall_row = choose_best(rows)
    best_adam_row = choose_best(solver_rows(rows, "adam"))
    best_nesterov_row = choose_best(solver_rows(rows, "nesterov"))

    best_overall_candidate = exp0.row_to_candidate(best_overall_row, candidates)
    best_adam_candidate = exp0.row_to_candidate(best_adam_row, candidates)
    best_nesterov_candidate = exp0.row_to_candidate(best_nesterov_row, candidates)

    exp0.save_best_hyperparameters(
        best_overall_candidate,
        best_overall_row,
        config.run_path / "exp0_best_hyperparameters.json",
        config=config,
    )

    payload = {
        "best_overall": exp0.best_payload(best_overall_candidate, best_overall_row),
        "best_adam": exp0.best_payload(best_adam_candidate, best_adam_row),
        "best_nesterov": exp0.best_payload(best_nesterov_candidate, best_nesterov_row),
    }
    (config.run_path / "exp0_best_by_solver_hyperparameters.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return best_adam_candidate, best_nesterov_candidate


def run_solver_comparison_curve(config, train_partitions, validation_data, selected_writer_ids, adam_candidate, nesterov_candidate):
    final_path = config.run_path / "final_solver_comparison_curve"
    final_path.mkdir(parents=True, exist_ok=True)

    state_snapshot_period = max(1, config.final_iterations // 10)
    problem, x0 = exp0.build_problem(
        train_partitions,
        validation_data,
        selected_writer_ids,
        state_snapshot_period=state_snapshot_period,
    )

    adam_algorithm = exp0.build_algorithm(adam_candidate, x0, config.final_iterations)
    nesterov_algorithm = exp0.build_algorithm(nesterov_candidate, x0, config.final_iterations)
    adam_algorithm.name = "FedLT-Adam"
    nesterov_algorithm.name = "FedLT-Nesterov"

    result = exp0.benchmark.benchmark(
        algorithms=[adam_algorithm, nesterov_algorithm],
        benchmark_problem=problem,
        n_trials=config.n_trials,
        max_processes=1,
        progress_step=max(1, config.final_iterations // 10),
        show_speed=True,
        show_trial=True,
        checkpoint_manager=None,
        log_level=logging.INFO,
    )

    metric_result = exp0.benchmark.compute_metrics(
        benchmark_result=result,
        table_metrics=[
            exp0.ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
            exp0.ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
            exp0.ml.Loss([np.average]),
        ],
        plot_metrics=[
            exp0.ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
            exp0.ml.Loss([np.average]),
        ],
        log_level=logging.INFO,
    )

    exp0.benchmark.display_metrics(
        metrics_result=metric_result,
        save_path=final_path / "results",
        show_plots=False,
        log_level=logging.INFO,
    )

    metric_result.agent_metrics = None
    exp0.save_pickle_zst(metric_result, final_path / exp0.metric_result_filename)
    (final_path / "metric_computation_complete.json").write_text(
        json.dumps({"metric_computation_complete": True}, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "final_iterations": config.final_iterations,
        "state_snapshot_period": state_snapshot_period,
        "n_trials": config.n_trials,
        "best_adam_candidate": exp0.best_payload(adam_candidate, row=None),
        "best_nesterov_candidate": exp0.best_payload(nesterov_candidate, row=None),
    }
    (final_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_name = f"run_fedlt_refined_adam_vs_nesterov_{datetime.now():%Y%m%d_%H%M%S}"
    config = exp0.RuntimeConfig(
        algorithm="fedlt",
        iterations=ITERATIONS,
        final_iterations=FINAL_ITERATIONS,
        n_trials=N_TRIALS,
        n_random_candidates=0,
        max_grid_candidates=0,
        run_final=True,
        combined_curves=False,
        run_path=Path("experiments/femnist/checkpoints/experiment0/fedlt") / run_name,
    )
    config.run_path.mkdir(parents=True, exist_ok=True)

    train_partitions, validation_data, selected_writer_ids = exp0.load_data()
    candidates = build_candidates()
    print(f"Running {len(candidates)} FedLT candidates")
    save_dataset_metadata(config, selected_writer_ids, validation_data, train_partitions, len(candidates))

    rows = exp0.run_candidate_list(
        candidates,
        train_partitions,
        validation_data,
        selected_writer_ids,
        config=config,
        candidate_results_path=config.run_path / "exp0_candidate_results.csv",
        starting_index=1,
    )

    best_adam_candidate, best_nesterov_candidate = save_best_by_solver(config, candidates, rows)
    run_solver_comparison_curve(
        config,
        train_partitions,
        validation_data,
        selected_writer_ids,
        best_adam_candidate,
        best_nesterov_candidate,
    )

    print(f"FedLT refined tuning complete: {config.run_path}")


if __name__ == "__main__":
    main()
