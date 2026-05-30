"""Stability-focused FedLT tuning for FEMNIST Experiment 0."""

from __future__ import annotations

import json
import math
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import experiment0 as exp0


ITERATIONS = 1500
FINAL_ITERATIONS = 1500
N_TRIALS = 3

ADAM_STEP_SIZE_VALUES = [0.003, 0.005, 0.0075]
ADAM_NUM_LOCAL_EPOCHS_VALUES = [5]
ADAM_RHO_VALUES = [0.1, 1.0]
ADAM_BETA1_VALUES = [0.5, 0.9]
ADAM_BETA2_VALUES = [0.999]
ADAM_EPSILON = 1e-8

GD_STEP_SIZE_VALUES = [0.01, 0.02]
GD_NUM_LOCAL_EPOCHS_VALUES = [3, 5]
GD_RHO_VALUES = [1.0]


def fedlt_candidate(
    *,
    variant: str,
    search_stage: str,
    step_size: float,
    num_local_epochs: int,
    rho: float,
    local_solver: str,
    solver_args: dict[str, float] | None = None,
) -> exp0.Candidate:
    hyperparameters: dict[str, Any] = {
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
        search_stage=search_stage,
        hyperparameters=hyperparameters,
    )


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
                    f"adam_lr_{exp0.format_value(step_size)}_e{num_local_epochs}"
                    f"_rho_{exp0.format_value(rho)}_b1_{exp0.format_value(beta1)}"
                    f"_b2_{exp0.format_value(beta2)}"
                ),
                search_stage="stability_grid",
                step_size=step_size,
                num_local_epochs=num_local_epochs,
                rho=rho,
                local_solver="adam",
                solver_args={"beta1": beta1, "beta2": beta2, "epsilon": ADAM_EPSILON},
            )
        )

    for step_size, num_local_epochs, rho in product(
        GD_STEP_SIZE_VALUES,
        GD_NUM_LOCAL_EPOCHS_VALUES,
        GD_RHO_VALUES,
    ):
        candidates.append(
            fedlt_candidate(
                variant=(
                    f"gd_reference_lr_{exp0.format_value(step_size)}"
                    f"_e{num_local_epochs}_rho_{exp0.format_value(rho)}"
                ),
                search_stage="reference_grid",
                step_size=step_size,
                num_local_epochs=num_local_epochs,
                rho=rho,
                local_solver="gd",
            )
        )

    return exp0.deduplicate_candidates(candidates)


def finite_float(value: Any) -> float | None:
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    return float_value if math.isfinite(float_value) else None


def is_selectable(row: dict[str, Any]) -> bool:
    return (
        row["status"] == "ok"
        and finite_float(row.get("server_accuracy_mean")) is not None
        and finite_float(row.get("server_accuracy_margin_of_error")) is not None
        and finite_float(row.get("validation_loss_mean")) is not None
    )


def robust_score(row: dict[str, Any]) -> tuple[float, float]:
    accuracy = float(row["server_accuracy_mean"])
    accuracy_margin = float(row["server_accuracy_margin_of_error"])
    validation_loss = float(row["validation_loss_mean"])
    return accuracy - accuracy_margin, -validation_loss


def select_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selectable_rows = [row for row in rows if is_selectable(row)]
    if not selectable_rows:
        raise RuntimeError("No stable finite FedLT candidates were available.")
    return max(selectable_rows, key=robust_score)


def save_dataset_metadata(
    config: exp0.RuntimeConfig,
    selected_writer_ids: list[str],
    validation_data: exp0.Dataset,
    train_partitions: list[exp0.Dataset],
    candidate_count: int,
) -> None:
    payload = {
        "selected_writer_ids": list(selected_writer_ids),
        "n_validation_samples": len(validation_data),
        "n_train_samples_after_validation_split": sum(len(partition) for partition in train_partitions),
        "run_config": {
            "algorithm": "fedlt",
            "iterations": config.iterations,
            "final_iterations": config.final_iterations,
            "n_trials": config.n_trials,
            "run_final": config.run_final,
            "selection_fraction": exp0.selection_fraction,
            "candidate_count": candidate_count,
            "adam": {
                "step_size_values": ADAM_STEP_SIZE_VALUES,
                "num_local_epochs_values": ADAM_NUM_LOCAL_EPOCHS_VALUES,
                "rho_values": ADAM_RHO_VALUES,
                "beta1_values": ADAM_BETA1_VALUES,
                "beta2_values": ADAM_BETA2_VALUES,
                "epsilon": ADAM_EPSILON,
            },
            "gd_reference": {
                "step_size_values": GD_STEP_SIZE_VALUES,
                "num_local_epochs_values": GD_NUM_LOCAL_EPOCHS_VALUES,
                "rho_values": GD_RHO_VALUES,
            },
        },
        "selection_rule": (
            "Reject non-finite validation metrics. Rank remaining candidates by "
            "server accuracy minus its margin of error, then by validation loss."
        ),
        "notes": (
            "FedLT stability retuning after the 3-trial baseline showed high client drift. "
            "Focuses on the previously best Adam solver region and includes GD references."
        ),
    }
    (config.run_path / "exp0_dataset_metadata.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def save_best_hyperparameters(
    best_candidate: exp0.Candidate,
    best_row: dict[str, Any],
    config: exp0.RuntimeConfig,
) -> None:
    payload = {
        "metadata": {
            "experiment": "experiment0",
            "algorithm": "fedlt",
            "dataset": "FEMNIST",
            "dataset_source": "flwrlabs/femnist",
            "partition": "natural writer/client split",
            "n_clients": exp0.n_clients,
            "min_train_samples": exp0.min_train_samples,
            "min_test_samples": exp0.min_test_samples,
            "train_fraction": exp0.train_fraction,
            "validation_fraction_from_train": exp0.validation_fraction,
            "n_trials": config.n_trials,
            "iterations": config.iterations,
            "state_snapshot_period": config.iterations,
            "checkpoint_step": None,
            "batch_size": exp0.batch_size,
            "seed": exp0.seed,
            "selection_fraction": exp0.selection_fraction,
            "selection_metric": "server accuracy lower confidence bound",
            "tie_break_metric": "loss",
            "search_strategy": "focused deterministic stability grid",
        },
        "best_hyperparameters": exp0.best_payload(best_candidate, best_row),
    }
    (config.run_path / "exp0_best_hyperparameters.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    run_name = f"run_fedlt_stability_ntrials3_{datetime.now():%Y%m%d_%H%M%S}"
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

    best_result_row = select_best_row(rows)
    best_candidate = exp0.row_to_candidate(best_result_row, candidates)
    save_best_hyperparameters(best_candidate, best_result_row, config)
    exp0.run_final_curve(
        best_candidate,
        train_partitions,
        validation_data,
        selected_writer_ids,
        config=config,
    )

    print(f"FedLT stability tuning complete: {config.run_path}")


if __name__ == "__main__":
    main()
