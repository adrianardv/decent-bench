# ruff: noqa: D103, E402, INP001, T201
"""
FEMNIST aggregation-weighting benchmark on a high quantity-imbalance subset.

This repeats the clean Experiment 2 uniform-vs-data-size weighted aggregation
comparison, but replaces the fixed random FEMNIST writer subset with a
deterministic subset containing the smallest and largest eligible writers.
All other benchmark parameters are kept aligned with Experiment 2.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import decent_bench.utils.interoperability as iop
from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.utils import pytorch_initialization
from decent_bench.benchmark import MetricResult
from decent_bench.costs import PyTorchCost
from decent_bench.metrics import metric_library as ml
from decent_bench.networks import FedNetwork
from decent_bench.schemes import AlwaysActive, NoCompression, NoDrops, NoNoise
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.pytorch_utils import ArgmaxActivation
from experiments.femnist.experiment2 import experiment2_aggregation_weighting as clean_exp2
from experiments.femnist.src import FEMNISTCNN, FEMNISTDatasetHandler
from experiments.femnist.src.inspection_helpers import (
    FEMNIST_CLASS_NAMES,
    add_seeded_per_writer_train_test_split,
    client_stats,
    load_huggingface_metadata,
)

experiment_group = "experiment6"
experiment_name = "experiment6_quantity_imbalance_aggregation_weighting"
metric_result_filename = clean_exp2.metric_result_filename
algorithm_order = clean_exp2.algorithm_order
subset_name = "quantity_imbalance_extremes"
subset_artifact_dir = Path("experiments/femnist/experiment6")
low_quantity_clients = clean_exp2.n_clients // 2
high_quantity_clients = clean_exp2.n_clients - low_quantity_clients

algorithm_labels = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "SCAFFOLD",
    "fednova": "FedNova",
    "fedopt": "FedAdam",
    "fedlt": "FedLT",
    "feddyn": "FedDyn",
}


def build_plot_metrics() -> list[ml.Metric]:
    return [
        ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
        ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
        ml.ClientDriftFromServer([], x_log=False, y_log=False),
    ]


def select_quantity_imbalance_clients() -> pd.DataFrame:
    metadata = load_huggingface_metadata(
        Path("experiments/femnist/data/cache"),
        local_files_only=clean_exp2.local_files_only,
    )
    metadata = add_seeded_per_writer_train_test_split(
        metadata,
        train_fraction=clean_exp2.train_fraction,
        seed=clean_exp2.seed,
    )
    stats = client_stats(metadata)
    eligible = stats[
        (stats["train_samples"] >= clean_exp2.min_train_samples)
        & (stats["test_samples"] >= clean_exp2.min_test_samples)
    ].copy()

    if len(eligible) < clean_exp2.n_clients:
        raise ValueError(
            f"Requested {clean_exp2.n_clients} clients, but only {len(eligible)} satisfy "
            f"min_train_samples={clean_exp2.min_train_samples} and "
            f"min_test_samples={clean_exp2.min_test_samples}."
        )

    smallest = eligible.sort_values(["train_samples", "writer_id"], ascending=[True, True]).head(low_quantity_clients)
    largest = (
        eligible.drop(index=smallest.index)
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


def selected_label_totals(selected: pd.DataFrame) -> dict[int, int]:
    totals = dict.fromkeys(range(len(FEMNIST_CLASS_NAMES)), 0)
    for histogram in selected["label_histogram"]:
        parsed = cast("dict[str, int]", json.loads(str(histogram)))
        for label, count in parsed.items():
            totals[int(label)] += int(count)
    return totals


def save_selected_client_dataset_size_plot(selected: pd.DataFrame, output_path: Path) -> None:
    plotted = selected.sort_values(["train_samples", "writer_id"]).reset_index(drop=True)
    colors = np.where(plotted["quantity_group"] == "small", "#4C78A8", "#F58518")

    x_positions = np.arange(len(plotted))
    fig, ax = plt.subplots(figsize=(18, 6), layout="constrained")
    ax.bar(x_positions, plotted["train_samples"], color=colors)
    ax.set_xlabel("Writer ID")
    ax.set_ylabel("Train samples")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plotted["writer_id"].astype(str), rotation=90, fontsize=6)
    ax.grid(visible=True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlim(-0.75, len(plotted) - 0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_shared_selected_clients() -> tuple[Path, dict[str, Any]]:
    selected = select_quantity_imbalance_clients()
    subset_artifact_dir.mkdir(parents=True, exist_ok=True)

    selected_clients_path = subset_artifact_dir / f"{subset_name}_selected_clients.csv"
    selected.to_csv(selected_clients_path, index=False)
    dataset_size_plot_path = subset_artifact_dir / f"{subset_name}_train_dataset_sizes.png"
    save_selected_client_dataset_size_plot(selected, dataset_size_plot_path)

    label_totals = selected_label_totals(selected)
    missing_labels = [label for label, count in label_totals.items() if count == 0]
    summary = {
        "subset_name": subset_name,
        "selection_rule": (
            f"{low_quantity_clients} eligible writers with the fewest train samples and "
            f"{high_quantity_clients} eligible writers with the most train samples"
        ),
        "n_clients": len(selected),
        "min_train_samples_threshold": clean_exp2.min_train_samples,
        "min_test_samples_threshold": clean_exp2.min_test_samples,
        "train_sample_quantiles": {
            str(key): float(value)
            for key, value in selected["train_samples"].quantile([0, 0.25, 0.5, 0.75, 1.0]).items()
        },
        "test_sample_quantiles": {
            str(key): float(value)
            for key, value in selected["test_samples"].quantile([0, 0.25, 0.5, 0.75, 1.0]).items()
        },
        "n_classes_per_client_quantiles": {
            str(key): float(value) for key, value in selected["n_classes"].quantile([0, 0.25, 0.5, 0.75, 1.0]).items()
        },
        "train_sample_imbalance_ratio": float(selected["train_samples"].max() / selected["train_samples"].min()),
        "selected_total_train_samples": int(selected["train_samples"].sum()),
        "selected_total_test_samples": int(selected["test_samples"].sum()),
        "selected_classes_covered": int(len(FEMNIST_CLASS_NAMES) - len(missing_labels)),
        "selected_missing_classes": missing_labels,
        "selected_clients_path": str(selected_clients_path),
        "selected_client_dataset_size_plot": str(dataset_size_plot_path),
    }
    (subset_artifact_dir / "quantity_imbalance_subset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return selected_clients_path, summary


def build_problem(selected_clients_path: Path) -> tuple[benchmark.BenchmarkProblem, Any, list[str], int, int]:
    iop.set_seed(clean_exp2.seed)
    train_dataset = FEMNISTDatasetHandler(
        split="train",
        selected_clients_path=selected_clients_path,
        n_clients=clean_exp2.n_clients,
        train_fraction=clean_exp2.train_fraction,
        seed=clean_exp2.seed,
        min_train_samples=clean_exp2.min_train_samples,
        min_test_samples=clean_exp2.min_test_samples,
        local_files_only=clean_exp2.local_files_only,
    )
    test_dataset = FEMNISTDatasetHandler(
        split="test",
        selected_clients_path=selected_clients_path,
        n_clients=clean_exp2.n_clients,
        train_fraction=clean_exp2.train_fraction,
        seed=clean_exp2.seed,
        min_train_samples=clean_exp2.min_train_samples,
        min_test_samples=clean_exp2.min_test_samples,
        local_files_only=clean_exp2.local_files_only,
    )

    train_partitions = train_dataset.get_partitions()
    costs = [
        PyTorchCost(
            dataset=partition,
            model=FEMNISTCNN(),
            loss_fn=torch.nn.CrossEntropyLoss(),
            final_activation=ArgmaxActivation(),
            batch_size=min(clean_exp2.batch_size, len(partition)),
            device=clean_exp2.device,
            load_dataset=clean_exp2.load_dataset,
        )
        for partition in train_partitions
    ]
    agents = [
        Agent(
            cost,
            activation=AlwaysActive(),
            state_snapshot_period=clean_exp2.state_snapshot_period,
            data={"writer_id": writer_id},
        )
        for writer_id, cost in zip(train_dataset.selected_writer_ids, costs, strict=True)
    ]
    network = FedNetwork(
        clients=agents,
        message_noise=NoNoise(),
        message_compression=NoCompression(),
        message_drop=NoDrops(),
    )
    problem = benchmark.BenchmarkProblem(
        network=network,
        test_data=test_dataset.get_datapoints(),
    )
    x0 = pytorch_initialization(network, all_same=True)
    return (
        problem,
        x0,
        list(train_dataset.selected_writer_ids),
        sum(len(partition) for partition in train_partitions),
        len(test_dataset.get_datapoints()),
    )


def aggregation_label(algorithm_name: str) -> str:
    if "data-size weighted" in algorithm_name:
        return "data-size weighted"
    if "uniform" in algorithm_name:
        return "uniform"
    raise ValueError(f"Cannot infer aggregation label from algorithm name: {algorithm_name}")


def build_server_accuracy_table(table_frame: pd.DataFrame, algorithm_key: str) -> pd.DataFrame:
    server_accuracy = table_frame[(table_frame["metric"] == "server accuracy") & (table_frame["statistic"].isna())]
    if server_accuracy.empty:
        server_accuracy = table_frame[(table_frame["metric"] == "server accuracy") & (table_frame["statistic"] == "")]

    rows: dict[str, float] = {}
    margins: dict[str, float] = {}
    for _, row in server_accuracy.iterrows():
        column = aggregation_label(str(row["algorithm"]))
        rows[column] = float(row["mean"])
        margins[f"{column} margin_of_error"] = float(row["margin_of_error"])

    return pd.DataFrame(
        [
            {
                "algorithm": algorithm_labels[algorithm_key],
                "uniform": rows.get("uniform", float("nan")),
                "data-size weighted": rows.get("data-size weighted", float("nan")),
                "uniform margin_of_error": margins.get("uniform margin_of_error", float("nan")),
                "data-size weighted margin_of_error": margins.get("data-size weighted margin_of_error", float("nan")),
            }
        ]
    )


def write_server_accuracy_table(metric_result: MetricResult, algorithm_key: str, output_dir: Path) -> None:
    table_frame, _ = metric_result.to_dataframe()
    if table_frame is None:
        raise ValueError("No table metrics available for server-accuracy summary.")

    server_table = build_server_accuracy_table(table_frame, algorithm_key)
    server_table.to_csv(output_dir / "server_accuracy_by_aggregation.csv", index=False)

    formatted = server_table.copy()
    for column in ("uniform", "data-size weighted", "uniform margin_of_error", "data-size weighted margin_of_error"):
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    formatted.to_markdown(str(output_dir / "server_accuracy_by_aggregation.md"), index=False)
    formatted.to_latex(str(output_dir / "server_accuracy_by_aggregation.tex"), index=False)


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
        "title": "FEMNIST aggregation weighting on a high quantity-imbalance subset",
        "purpose": (
            "Repeat clean Experiment 2's uniform-vs-data-size weighted aggregation comparison while selecting "
            "writers with strongly different local dataset sizes."
        ),
        "execution": (
            "one algorithm pair per benchmark() call; each pair writes to experiment6/<algorithm>/run_<timestamp>"
        ),
        "dataset": "FEMNIST",
        "dataset_source": "flwrlabs/femnist",
        "partition": "natural writer/client split",
        "subset": subset_summary,
        "selected_clients_path": str(selected_clients_path),
        "n_clients": clean_exp2.n_clients,
        "min_train_samples": clean_exp2.min_train_samples,
        "min_test_samples": clean_exp2.min_test_samples,
        "n_classes": 62,
        "train_fraction": clean_exp2.train_fraction,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "n_trials": clean_exp2.n_trials,
        "iterations": iterations,
        "state_snapshot_period": state_snapshot_period,
        "checkpoint_step": clean_exp2.checkpoint_step,
        "batch_size": clean_exp2.batch_size,
        "seed": clean_exp2.seed,
        "device": str(clean_exp2.device),
        "load_dataset": clean_exp2.load_dataset,
        "selected_hyperparameters_path": str(clean_exp2.selected_hyperparameters_path),
        "model": "CNN: conv 1->32, conv 32->64, dense 256, output 62 logits",
        "loss": "torch.nn.CrossEntropyLoss",
        "network": {
            "participation": "partial",
            "client_selection": "UniformSelection",
            "selection_fraction": clean_exp2.selection_fraction,
            "clients_per_round": int(clean_exp2.n_clients * clean_exp2.selection_fraction),
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
    (run_path / "experiment6_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_algorithm_pair(
    *,
    algorithm_key: str,
    selected_hyperparameters: dict[str, Any],
    run_path: Path,
    selected_clients_path: Path,
    statuses: list[dict[str, Any]],
    iterations: int,
    state_snapshot_period: int,
) -> tuple[list[str], int, int]:
    print(f"Running {algorithm_key} on {subset_name}; results: {run_path}")

    problem, x0, writer_ids, n_train_samples, n_test_samples = build_problem(selected_clients_path)
    algorithms = clean_exp2.build_algorithms(algorithm_key, x0, selected_hyperparameters)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=run_path,
        checkpoint_step=clean_exp2.checkpoint_step,
        keep_n_checkpoints=1,
        benchmark_metadata={
            "experiment": experiment_name,
            "subset": subset_name,
            "algorithm": algorithm_key,
            "aggregation_variants": [algorithm.name for algorithm in algorithms],
            "n_trials": clean_exp2.n_trials,
            "iterations": iterations,
            "state_snapshot_period": state_snapshot_period,
            "checkpoint_step": clean_exp2.checkpoint_step,
        },
    )

    try:
        result = benchmark.benchmark(
            algorithms=algorithms,
            benchmark_problem=problem,
            n_trials=clean_exp2.n_trials,
            max_processes=1,
            progress_step=clean_exp2.progress_step,
            show_speed=True,
            show_trial=True,
            checkpoint_manager=checkpoint_manager,
            log_level=logging.INFO,
        )
        metric_result = benchmark.compute_metrics(
            benchmark_result=result,
            table_metrics=clean_exp2.build_table_metrics(),
            plot_metrics=build_plot_metrics(),
            checkpoint_manager=checkpoint_manager,
            log_level=logging.INFO,
        )
        benchmark.display_metrics(
            metrics_result=metric_result,
            checkpoint_manager=checkpoint_manager,
            individual_plots=True,
            show_plots=clean_exp2.show_plots,
            log_level=logging.INFO,
        )
        results_path = checkpoint_manager.get_results_path()
        clean_exp2.save_metric_dataframes(metric_result, results_path)
        write_server_accuracy_table(metric_result, algorithm_key, results_path)
        metric_result.agent_metrics = None
        clean_exp2.save_pickle_zst(metric_result, run_path / metric_result_filename)
        (run_path / "metric_computation_complete.json").write_text(
            json.dumps({"metric_computation_complete": True}, indent=2),
            encoding="utf-8",
        )
        statuses.append(
            {
                "algorithm": algorithm_key,
                "status": "ok",
                "variants": [algorithm.name for algorithm in algorithms],
            }
        )

    except Exception as error:
        statuses.append(
            {
                "algorithm": algorithm_key,
                "status": "failed",
                "variants": [algorithm.name for algorithm in algorithms],
                "error": repr(error),
            }
        )
        raise

    else:
        return writer_ids, n_train_samples, n_test_samples

    finally:
        if "metric_result" in locals():
            del metric_result
        if "result" in locals():
            del result
        del algorithms, problem, x0
        clean_exp2.clear_cuda_cache()


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
        default=clean_exp2.iterations,
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
        state_snapshot_period = clean_exp2.state_snapshot_period
        if args.iterations != clean_exp2.iterations:
            state_snapshot_period = max(1, args.iterations // 10)
    if state_snapshot_period <= 0:
        raise ValueError("--state-snapshot-period must be positive")

    clean_exp2.iterations = args.iterations
    clean_exp2.state_snapshot_period = state_snapshot_period

    run_id = f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    if args.run_label:
        run_id = f"{run_id}_{clean_exp2.slugify(args.run_label)}"
    checkpoint_root = Path("experiments/femnist/checkpoints") / experiment_group
    selected_hyperparameters = clean_exp2.load_selected_hyperparameters()

    print(f"Writing Experiment 6 results under: {checkpoint_root}")
    selected_clients_path, subset_summary = write_shared_selected_clients()

    for algorithm_key in requested_algorithms:
        run_path = checkpoint_root / algorithm_key / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        selected_writer_ids: list[str] = []
        n_train_samples = 0
        n_test_samples = 0
        statuses: list[dict[str, Any]] = []

        try:
            selected_writer_ids, n_train_samples, n_test_samples = run_algorithm_pair(
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

    print(f"Experiment 6 quantity-imbalance aggregation-weighting benchmark complete: {checkpoint_root}")


if __name__ == "__main__":
    main()
