# ruff: noqa: D103, E402, INP001, T201
"""
FEMNIST aggregation-weighting benchmark on the maximum quantity-imbalance subset.

This repeats Experiment 6's clean uniform-vs-data-size weighted aggregation
comparison, but removes the minimum train/test sample eligibility thresholds
from the subset selection. The selected subset contains the 50 FEMNIST writers
with the fewest train samples and the 50 writers with the most train samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.femnist.experiment6 import experiment6_quantity_imbalance_aggregation_weighting as base_exp6

experiment_group = "experiment7"
experiment_name = "experiment7_unbounded_quantity_imbalance_aggregation_weighting"
subset_name = "unbounded_quantity_imbalance_extremes"
subset_artifact_dir = Path("experiments/femnist/experiment7")
low_quantity_clients = base_exp6.clean_exp2.n_clients // 2
high_quantity_clients = base_exp6.clean_exp2.n_clients - low_quantity_clients
metric_result_filename = base_exp6.metric_result_filename
algorithm_order = base_exp6.algorithm_order

base_exp6.experiment_group = experiment_group
base_exp6.experiment_name = experiment_name
base_exp6.subset_name = subset_name
base_exp6.subset_artifact_dir = subset_artifact_dir


def select_unbounded_quantity_imbalance_clients() -> pd.DataFrame:
    metadata = base_exp6.load_huggingface_metadata(
        Path("experiments/femnist/data/cache"),
        local_files_only=base_exp6.clean_exp2.local_files_only,
    )
    metadata = base_exp6.add_seeded_per_writer_train_test_split(
        metadata,
        train_fraction=base_exp6.clean_exp2.train_fraction,
        seed=base_exp6.clean_exp2.seed,
    )
    stats = base_exp6.client_stats(metadata)

    if len(stats) < base_exp6.clean_exp2.n_clients:
        raise ValueError(f"Requested {base_exp6.clean_exp2.n_clients} clients, but only {len(stats)} are available.")

    smallest = stats.sort_values(["train_samples", "writer_id"], ascending=[True, True]).head(low_quantity_clients)
    largest = (
        stats.drop(index=smallest.index)
        .sort_values(["train_samples", "writer_id"], ascending=[False, True])
        .head(high_quantity_clients)
    )
    selected = pd.concat([smallest, largest], ignore_index=True)
    selected = selected.sort_values("writer_id").reset_index(drop=True)
    selected.insert(0, "client_index", np.arange(len(selected), dtype=np.int64))
    selected.insert(
        1,
        "quantity_group",
        np.where(selected["train_samples"] <= smallest["train_samples"].max(), "small", "large"),
    )
    return selected


def write_shared_selected_clients() -> tuple[Path, dict[str, Any]]:
    selected = select_unbounded_quantity_imbalance_clients()
    subset_artifact_dir.mkdir(parents=True, exist_ok=True)

    selected_clients_path = subset_artifact_dir / f"{subset_name}_selected_clients.csv"
    selected.to_csv(selected_clients_path, index=False)
    dataset_size_plot_path = subset_artifact_dir / f"{subset_name}_train_dataset_sizes_ids.png"
    base_exp6.save_selected_client_dataset_size_plot(selected, dataset_size_plot_path)

    label_totals = base_exp6.selected_label_totals(selected)
    missing_labels = [label for label, count in label_totals.items() if count == 0]
    summary = {
        "subset_name": subset_name,
        "selection_rule": (
            f"{low_quantity_clients} writers with the fewest train samples and "
            f"{high_quantity_clients} writers with the most train samples, without minimum train/test filtering"
        ),
        "selection_filter": "none",
        "n_clients": len(selected),
        "min_train_samples_threshold": None,
        "min_test_samples_threshold": None,
        "train_sample_quantiles": {
            str(key): float(value)
            for key, value in selected["train_samples"].quantile([0, 0.25, 0.5, 0.75, 1.0]).items()
        },
        "test_sample_quantiles": {
            str(key): float(value)
            for key, value in selected["test_samples"].quantile([0, 0.25, 0.5, 0.75, 1.0]).items()
        },
        "total_sample_quantiles": {
            str(key): float(value)
            for key, value in selected["total_samples"].quantile([0, 0.25, 0.5, 0.75, 1.0]).items()
        },
        "n_classes_per_client_quantiles": {
            str(key): float(value) for key, value in selected["n_classes"].quantile([0, 0.25, 0.5, 0.75, 1.0]).items()
        },
        "train_sample_imbalance_ratio": float(selected["train_samples"].max() / selected["train_samples"].min()),
        "total_sample_imbalance_ratio": float(selected["total_samples"].max() / selected["total_samples"].min()),
        "selected_total_train_samples": int(selected["train_samples"].sum()),
        "selected_total_test_samples": int(selected["test_samples"].sum()),
        "selected_total_samples": int(selected["total_samples"].sum()),
        "selected_classes_covered": int(len(base_exp6.FEMNIST_CLASS_NAMES) - len(missing_labels)),
        "selected_missing_classes": missing_labels,
        "selected_clients_path": str(selected_clients_path),
        "selected_client_dataset_size_plot": str(dataset_size_plot_path),
    }
    (subset_artifact_dir / "unbounded_quantity_imbalance_subset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return selected_clients_path, summary


def write_run_inputs(
    *,
    run_path: Path,
    selected_writer_ids: list[str],
    selected_clients_path: Path,
    subset_summary: dict[str, Any],
    n_train_samples: int,
    n_test_samples: int,
    requested_algorithms: list[str],
    statuses: list[dict[str, Any]],
    iterations: int,
    state_snapshot_period: int,
) -> None:
    run_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": experiment_name,
        "title": "FEMNIST aggregation weighting on a maximum quantity-imbalance subset",
        "purpose": (
            "Repeat Experiment 6's uniform-vs-data-size weighted aggregation comparison after removing the "
            "minimum train/test sample selection thresholds."
        ),
        "execution": (
            "one algorithm pair per benchmark() call; each pair writes to experiment7/<algorithm>/run_<timestamp>"
        ),
        "dataset": "FEMNIST",
        "dataset_source": "flwrlabs/femnist",
        "partition": "natural writer/client split",
        "subset": subset_summary,
        "selected_clients_path": str(selected_clients_path),
        "n_clients": base_exp6.clean_exp2.n_clients,
        "selection_min_train_samples": None,
        "selection_min_test_samples": None,
        "n_classes": 62,
        "train_fraction": base_exp6.clean_exp2.train_fraction,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "n_trials": base_exp6.clean_exp2.n_trials,
        "iterations": iterations,
        "state_snapshot_period": state_snapshot_period,
        "checkpoint_step": base_exp6.clean_exp2.checkpoint_step,
        "batch_size": base_exp6.clean_exp2.batch_size,
        "seed": base_exp6.clean_exp2.seed,
        "device": str(base_exp6.clean_exp2.device),
        "load_dataset": base_exp6.clean_exp2.load_dataset,
        "selected_hyperparameters_path": str(base_exp6.clean_exp2.selected_hyperparameters_path),
        "model": "CNN: conv 1->32, conv 32->64, dense 256, output 62 logits",
        "loss": "torch.nn.CrossEntropyLoss",
        "network": {
            "participation": "partial",
            "client_selection": "UniformSelection",
            "selection_fraction": base_exp6.clean_exp2.selection_fraction,
            "clients_per_round": int(base_exp6.clean_exp2.n_clients * base_exp6.clean_exp2.selection_fraction),
            "activation": "AlwaysActive",
            "drops": "NoDrops",
            "noise": "NoNoise",
            "compression": "NoCompression",
        },
        "aggregation_variants": {
            "uniform": "Each received client upload has equal aggregation weight.",
            "data-size weighted": "Each received client upload is weighted by its local training sample count.",
        },
        "requested_algorithms": requested_algorithms,
        "algorithms": list(algorithm_order),
        "selected_writer_ids": selected_writer_ids,
        "statuses": statuses,
    }
    (run_path / "experiment7_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default=None, choices=["all", *algorithm_order])
    parser.add_argument("--all", action="store_const", const="all", dest="algorithm_flag")
    for algorithm_key in algorithm_order:
        parser.add_argument(f"--{algorithm_key}", action="store_const", const=algorithm_key, dest="algorithm_flag")
    parser.add_argument("--run-label", default=None, help="Optional suffix for the output run directory.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=base_exp6.clean_exp2.iterations,
        help="Number of iterations for each aggregation variant.",
    )
    parser.add_argument(
        "--state-snapshot-period",
        type=int,
        default=None,
        help="State snapshot period. Defaults to Experiment 2's value, or iterations // 10 when iterations changes.",
    )
    return parser.parse_args()


def requested_algorithms_from_args(args: argparse.Namespace) -> list[str]:
    requested = args.algorithm_flag or args.algorithm or "all"
    if args.algorithm is not None and args.algorithm_flag is not None and args.algorithm != args.algorithm_flag:
        raise ValueError("Use either --algorithm or one algorithm flag, not both.")
    return list(algorithm_order) if requested == "all" else [requested]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    requested_algorithms = requested_algorithms_from_args(args)
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    state_snapshot_period = args.state_snapshot_period
    if state_snapshot_period is None:
        state_snapshot_period = base_exp6.clean_exp2.state_snapshot_period
        if args.iterations != base_exp6.clean_exp2.iterations:
            state_snapshot_period = max(1, args.iterations // 10)
    if state_snapshot_period <= 0:
        raise ValueError("--state-snapshot-period must be positive")

    base_exp6.clean_exp2.iterations = args.iterations
    base_exp6.clean_exp2.state_snapshot_period = state_snapshot_period

    run_id = f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    if args.run_label:
        run_id = f"{run_id}_{base_exp6.clean_exp2.slugify(args.run_label)}"
    checkpoint_root = Path("experiments/femnist/checkpoints") / experiment_group
    selected_hyperparameters = base_exp6.clean_exp2.load_selected_hyperparameters()

    print(f"Writing Experiment 7 results under: {checkpoint_root}")
    selected_clients_path, subset_summary = write_shared_selected_clients()

    for algorithm_key in requested_algorithms:
        run_path = checkpoint_root / algorithm_key / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        selected_writer_ids: list[str] = []
        n_train_samples = 0
        n_test_samples = 0
        statuses: list[dict[str, Any]] = []

        try:
            selected_writer_ids, n_train_samples, n_test_samples = base_exp6.run_algorithm_pair(
                algorithm_key=algorithm_key,
                selected_hyperparameters=selected_hyperparameters,
                run_path=run_path,
                selected_clients_path=selected_clients_path,
                statuses=statuses,
                iterations=args.iterations,
                state_snapshot_period=state_snapshot_period,
            )
        finally:
            write_run_inputs(
                run_path=run_path,
                selected_writer_ids=selected_writer_ids,
                selected_clients_path=selected_clients_path,
                subset_summary=subset_summary,
                n_train_samples=n_train_samples,
                n_test_samples=n_test_samples,
                requested_algorithms=[algorithm_key],
                statuses=statuses,
                iterations=args.iterations,
                state_snapshot_period=state_snapshot_period,
            )

    print(f"Experiment 7 maximum quantity-imbalance aggregation-weighting benchmark complete: {checkpoint_root}")


if __name__ == "__main__":
    main()
