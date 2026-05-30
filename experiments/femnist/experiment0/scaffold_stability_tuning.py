"""Stability-focused SCAFFOLD tuning for FEMNIST Experiment 0."""

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

STEP_SIZE_VALUES = [0.005, 0.01, 0.02]
NUM_LOCAL_EPOCHS_VALUES = [3, 5]
SERVER_STEP_SIZE_VALUES = [0.5, 1.0]


def scaffold_candidate(step_size: float, num_local_epochs: int, server_step_size: float) -> exp0.Candidate:
    return exp0.Candidate(
        algorithm_key="scaffold",
        group="SCAFFOLD",
        algorithm_name="Scaffold",
        variant=(
            f"stability_grid_lr_{exp0.format_value(step_size)}"
            f"_e{num_local_epochs}_server_{exp0.format_value(server_step_size)}"
        ),
        search_stage="stability_grid",
        hyperparameters={
            "step_size": step_size,
            "num_local_epochs": num_local_epochs,
            "server_step_size": server_step_size,
        },
    )


def build_candidates() -> list[exp0.Candidate]:
    candidates = [
        scaffold_candidate(step_size, num_local_epochs, server_step_size)
        for step_size, num_local_epochs, server_step_size in product(
            STEP_SIZE_VALUES,
            NUM_LOCAL_EPOCHS_VALUES,
            SERVER_STEP_SIZE_VALUES,
        )
    ]
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
        raise RuntimeError("No stable finite SCAFFOLD candidates were available.")
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
            "algorithm": "scaffold",
            "iterations": config.iterations,
            "final_iterations": config.final_iterations,
            "n_trials": config.n_trials,
            "run_final": config.run_final,
            "selection_fraction": exp0.selection_fraction,
            "candidate_count": candidate_count,
            "step_size_values": STEP_SIZE_VALUES,
            "num_local_epochs_values": NUM_LOCAL_EPOCHS_VALUES,
            "server_step_size_values": SERVER_STEP_SIZE_VALUES,
        },
        "selection_rule": (
            "Reject non-finite validation metrics. Rank remaining candidates by "
            "server accuracy minus its margin of error, then by validation loss."
        ),
        "notes": (
            "SCAFFOLD stability retuning after the 3-trial baseline showed that "
            "the previous n_trials=1 choice could collapse in one trial."
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
            "algorithm": "scaffold",
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
    run_name = f"run_scaffold_stability_ntrials3_{datetime.now():%Y%m%d_%H%M%S}"
    config = exp0.RuntimeConfig(
        algorithm="scaffold",
        iterations=ITERATIONS,
        final_iterations=FINAL_ITERATIONS,
        n_trials=N_TRIALS,
        n_random_candidates=0,
        max_grid_candidates=0,
        run_final=True,
        combined_curves=False,
        run_path=Path("experiments/femnist/checkpoints/experiment0/scaffold") / run_name,
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

    print(f"SCAFFOLD stability tuning complete: {config.run_path}")


if __name__ == "__main__":
    main()
