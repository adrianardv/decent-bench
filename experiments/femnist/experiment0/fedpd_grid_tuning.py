from __future__ import annotations

import json
from datetime import datetime
from itertools import product
from pathlib import Path

import experiment0 as exp0


STEP_SIZE_VALUES = [0.01, 0.02, 0.03]
LOCAL_EPOCH_VALUES = [3, 5]
ETA_VALUES = [0.3, 0.5, 1.0]
SKIP_PROBABILITY_VALUES = [0.2, 0.5]

ITERATIONS = 1000
FINAL_ITERATIONS = 2000
N_TRIALS = 1


def make_candidate(step_size: float, epochs: int, eta: float, skip_probability: float) -> exp0.Candidate:
    variant = (
        f"grid_lr_{exp0.format_value(step_size)}"
        f"_e{epochs}"
        f"_eta_{exp0.format_value(eta)}"
        f"_skip_{exp0.format_value(skip_probability)}"
    )
    return exp0.candidate(
        "fedpd",
        "FedPD",
        "FedPD",
        variant,
        "grid",
        {
            "step_size": step_size,
            "num_local_epochs": epochs,
            "eta": eta,
            "skip_probability": skip_probability,
        },
    )


def main() -> None:
    run_name = f"run_fedpd_grid36_{datetime.now():%Y%m%d_%H%M%S}"
    run_path = Path("experiments/femnist/checkpoints/experiment0/fedpd") / run_name

    config = exp0.RuntimeConfig(
        algorithm="fedpd",
        iterations=ITERATIONS,
        final_iterations=FINAL_ITERATIONS,
        n_trials=N_TRIALS,
        n_random_candidates=0,
        max_grid_candidates=0,
        run_final=True,
        combined_curves=False,
        run_path=run_path,
    )

    run_path.mkdir(parents=True, exist_ok=True)
    candidate_results_path = run_path / "exp0_candidate_results.csv"
    best_path = run_path / "exp0_best_hyperparameters.json"

    train_partitions, validation_data, selected_writer_ids = exp0.load_data()

    candidates = [
        make_candidate(step_size, epochs, eta, skip_probability)
        for step_size, epochs, eta, skip_probability in product(
            STEP_SIZE_VALUES,
            LOCAL_EPOCH_VALUES,
            ETA_VALUES,
            SKIP_PROBABILITY_VALUES,
        )
    ]

    metadata = {
        "selected_writer_ids": selected_writer_ids,
        "n_validation_samples": len(validation_data),
        "n_train_samples_after_validation_split": sum(len(partition) for partition in train_partitions),
        "run_config": {
            "algorithm": "fedpd",
            "iterations": ITERATIONS,
            "final_iterations": FINAL_ITERATIONS,
            "n_trials": N_TRIALS,
            "run_final": True,
            "selection_fraction": None,
            "candidate_count": len(candidates),
            "step_size_values": STEP_SIZE_VALUES,
            "num_local_epochs_values": LOCAL_EPOCH_VALUES,
            "eta_values": ETA_VALUES,
            "skip_probability_values": SKIP_PROBABILITY_VALUES,
        },
        "notes": (
            "Focused FedPD grid tuning run after the automatic best collapsed and the first manual stability check "
            "showed stable but imperfect loss behavior. Candidates use full participation because FedPD currently "
            "does not support partial client participation."
        ),
    }
    (run_path / "exp0_dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"FedPD focused grid tuning run path: {run_path}")
    print(f"Running {len(candidates)} grid candidates for {ITERATIONS} iterations each.")

    rows = exp0.run_candidate_list(
        candidates,
        train_partitions,
        validation_data,
        selected_writer_ids,
        config=config,
        candidate_results_path=candidate_results_path,
        starting_index=1,
    )

    best_row = exp0.best_row(rows)
    best_candidate = exp0.row_to_candidate(best_row, candidates)
    exp0.save_best_hyperparameters(best_candidate, best_row, best_path, config=config)

    print("Selected FedPD hyperparameters:")
    print(best_path.read_text(encoding="utf-8"))

    print(f"Running final {FINAL_ITERATIONS}-iteration curve.")
    exp0.run_final_curve(best_candidate, train_partitions, validation_data, selected_writer_ids, config=config)

    print(f"Done. Results saved in: {run_path}")


if __name__ == "__main__":
    main()
