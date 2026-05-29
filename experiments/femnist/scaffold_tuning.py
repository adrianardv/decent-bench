from __future__ import annotations

import json
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

from experiments.femnist import experiment0 as exp0


STEP_SIZE_LOWER = 1e-3
STEP_SIZE_UPPER = 5e-2
SERVER_STEP_LOWER = 0.5
SERVER_STEP_UPPER = 1.0
LOCAL_EPOCH_CHOICES = [1, 2, 3, 5, 8, 10, 20]

N_RANDOM_CANDIDATES = 8
MAX_GRID_CANDIDATES = 12
ITERATIONS = 1000
FINAL_ITERATIONS = 2000
N_TRIALS = 1


def make_candidate(variant: str, search_stage: str, params: dict) -> exp0.Candidate:
    return exp0.candidate("scaffold", "SCAFFOLD", "Scaffold", variant, search_stage, params)


def nearby_epoch_values(value: int) -> list[int]:
    closest = min(range(len(LOCAL_EPOCH_CHOICES)), key=lambda i: abs(LOCAL_EPOCH_CHOICES[i] - value))
    return LOCAL_EPOCH_CHOICES[max(0, closest - 1) : min(len(LOCAL_EPOCH_CHOICES), closest + 2)]


def manual_candidate() -> exp0.Candidate:
    return make_candidate(
        "manual_lr_0p01_e5_server_1",
        "manual",
        {
            "step_size": 0.01,
            "num_local_epochs": 5,
            "server_step_size": 1.0,
        },
    )


def random_candidates(rng: np.random.Generator) -> list[exp0.Candidate]:
    candidates = [manual_candidate()]
    for index in range(N_RANDOM_CANDIDATES):
        params = {
            "step_size": exp0.log_uniform(rng, STEP_SIZE_LOWER, STEP_SIZE_UPPER),
            "num_local_epochs": exp0.random_choice(rng, LOCAL_EPOCH_CHOICES),
            "server_step_size": float(rng.uniform(SERVER_STEP_LOWER, SERVER_STEP_UPPER)),
        }
        candidates.append(make_candidate(f"random_{index:02d}", "random", params))
    return exp0.deduplicate_candidates(candidates)


def grid_candidates(best: exp0.Candidate) -> list[exp0.Candidate]:
    params = best.hyperparameters
    grid = product(
        exp0.nearby_log_values(float(params["step_size"]), lower=STEP_SIZE_LOWER, upper=STEP_SIZE_UPPER),
        nearby_epoch_values(int(params["num_local_epochs"])),
        exp0.nearby_linear_values(float(params["server_step_size"]), lower=SERVER_STEP_LOWER, upper=SERVER_STEP_UPPER),
    )

    candidates = []
    for step_size, epochs, server_step_size in grid:
        params = {
            "step_size": step_size,
            "num_local_epochs": epochs,
            "server_step_size": server_step_size,
        }
        variant = (
            f"grid_lr_{exp0.format_value(step_size)}"
            f"_e{epochs}"
            f"_server_{exp0.format_value(server_step_size)}"
        )
        candidates.append(make_candidate(variant, "grid", params))

    return exp0.deduplicate_candidates(candidates)


def main() -> None:
    run_name = f"run_scaffold_1000_final2000_{datetime.now():%Y%m%d_%H%M%S}"
    run_path = Path("experiments/femnist/checkpoints/experiment0/scaffold") / run_name

    config = exp0.RuntimeConfig(
        algorithm="scaffold",
        iterations=ITERATIONS,
        final_iterations=FINAL_ITERATIONS,
        n_trials=N_TRIALS,
        n_random_candidates=N_RANDOM_CANDIDATES,
        max_grid_candidates=MAX_GRID_CANDIDATES,
        run_final=True,
        combined_curves=False,
        run_path=run_path,
    )

    run_path.mkdir(parents=True, exist_ok=True)
    candidate_results_path = run_path / "exp0_candidate_results.csv"
    best_path = run_path / "exp0_best_hyperparameters.json"

    train_partitions, validation_data, selected_writer_ids = exp0.load_data()

    metadata = {
        "selected_writer_ids": selected_writer_ids,
        "n_validation_samples": len(validation_data),
        "n_train_samples_after_validation_split": sum(len(partition) for partition in train_partitions),
        "run_config": {
            "algorithm": "scaffold",
            "iterations": ITERATIONS,
            "final_iterations": FINAL_ITERATIONS,
            "n_trials": N_TRIALS,
            "n_random_candidates": N_RANDOM_CANDIDATES,
            "max_grid_candidates": MAX_GRID_CANDIDATES,
            "selection_fraction": exp0.selection_fraction,
            "step_size_range": [STEP_SIZE_LOWER, STEP_SIZE_UPPER],
            "local_epoch_choices": LOCAL_EPOCH_CHOICES,
            "server_step_size_range": [SERVER_STEP_LOWER, SERVER_STEP_UPPER],
            "manual_candidate": {
                "step_size": 0.01,
                "num_local_epochs": 5,
                "server_step_size": 1.0,
            },
        },
        "notes": (
            "Dedicated SCAFFOLD rerun after the generic Experiment 0 selected a configuration "
            "that looked good at 1000 iterations but collapsed during the 2000-iteration final curve. "
            "Candidates are tuned for 1000 iterations and the selected candidate is checked with a "
            "2000-iteration final curve."
        ),
    }
    (run_path / "exp0_dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rng = np.random.default_rng(exp0.seed)

    coarse_candidates = random_candidates(rng)
    print(f"SCAFFOLD tuning run path: {run_path}")
    print(f"Running {len(coarse_candidates)} manual/random candidates for {ITERATIONS} iterations.")

    all_candidates = list(coarse_candidates)
    all_rows = exp0.run_candidate_list(
        coarse_candidates,
        train_partitions,
        validation_data,
        selected_writer_ids,
        config=config,
        candidate_results_path=candidate_results_path,
        starting_index=1,
    )

    coarse_best = exp0.row_to_candidate(exp0.best_row(all_rows), coarse_candidates)
    focused_candidates = exp0.limit_grid_candidates(
        grid_candidates(coarse_best),
        max_candidates=MAX_GRID_CANDIDATES,
        rng=rng,
    )

    print(f"Running {len(focused_candidates)} focused grid candidates for {ITERATIONS} iterations.")

    all_candidates.extend(focused_candidates)
    grid_rows = exp0.run_candidate_list(
        focused_candidates,
        train_partitions,
        validation_data,
        selected_writer_ids,
        config=config,
        candidate_results_path=candidate_results_path,
        starting_index=len(all_rows) + 1,
    )
    all_rows.extend(grid_rows)

    best_row = exp0.best_row(all_rows)
    best_candidate = exp0.row_to_candidate(best_row, all_candidates)
    exp0.save_best_hyperparameters(best_candidate, best_row, best_path, config=config)

    print("Selected SCAFFOLD hyperparameters:")
    print(best_path.read_text(encoding="utf-8"))

    print(f"Running final {FINAL_ITERATIONS}-iteration curve.")
    exp0.run_final_curve(best_candidate, train_partitions, validation_data, selected_writer_ids, config=config)

    print(f"Done. Results saved in: {run_path}")


if __name__ == "__main__":
    main()
