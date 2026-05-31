# ruff: noqa: ANN401, D103, E402, INP001, PLR0911, T201
"""
FEMNIST aggregation-weighting benchmark.

For each tuned federated algorithm, this experiment compares uniform client
aggregation against data-size weighted aggregation using the same FEMNIST setup,
model, hyperparameters, and metrics as the other thesis experiments.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zstandard as zstd

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import decent_bench.utils.interoperability as iop
from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedAdam, FedAvg, FedDyn, FedLT, FedNova, FedProx, Scaffold
from decent_bench.algorithms.utils import pytorch_initialization
from decent_bench.benchmark import MetricResult
from decent_bench.costs import PyTorchCost
from decent_bench.metrics import metric_library as ml
from decent_bench.networks import FedNetwork
from decent_bench.schemes import AlwaysActive, NoCompression, NoDrops, NoNoise, UniformSelection
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.pytorch_utils import ArgmaxActivation
from decent_bench.utils.types import SupportedDevices
from experiments.femnist.src import FEMNISTCNN, FEMNISTDatasetHandler

# -----------------------------------------------------------------------------
# Fixed FEMNIST benchmark setup
# -----------------------------------------------------------------------------

seed = 20260524
n_clients = 100
min_train_samples = 100
min_test_samples = 20
train_fraction = 0.8
selection_fraction = 0.2
n_trials = 3
iterations = 1500
state_snapshot_period = 150
progress_step = 150
checkpoint_step = None
batch_size = 32
device = SupportedDevices.GPU
local_files_only = False
load_dataset = True
show_plots = False

experiment_group = "experiment2"
experiment_name = "experiment2_aggregation_weighting"
selected_hyperparameters_path = Path("experiments/femnist/experiment0/selected_hyperparameters.json")
metric_result_filename = "metric_computation.pkl.zst"

algorithm_order = (
    "fedavg",
    "fedprox",
    "scaffold",
    "fednova",
    "fedopt",
    "fedlt",
    "feddyn",
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def load_selected_hyperparameters() -> dict[str, Any]:
    with selected_hyperparameters_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def algorithm_hyperparameters(selected_hyperparameters: dict[str, Any], key: str) -> dict[str, Any]:
    return dict(selected_hyperparameters["algorithms"][key]["hyperparameters"])


def make_selection_scheme() -> UniformSelection:
    return UniformSelection(fraction_selected_clients=selection_fraction)


def save_pickle_zst(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = zstd.ZstdCompressor().compress(pickle.dumps(data))
    path.write_bytes(compressed)


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_table_metrics() -> list[ml.Metric]:
    return [
        ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
        ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
        ml.Loss([np.average]),
        ml.ClientDriftFromServer([min, np.average, max], x_log=False, y_log=False),
        ml.FractionSelectedClients(fmt=".2%", x_log=False, y_log=False),
        ml.GradientCalls([np.average, sum]),
        ml.SentMessages([np.average, sum]),
        ml.ReceivedMessages([np.average, sum]),
    ]


def build_plot_metrics() -> list[ml.Metric]:
    return [
        ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
        ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
        ml.Loss([np.average]),
        ml.ClientDriftFromServer([], x_log=False, y_log=False),
    ]


def metric_lookup(metrics: list[ml.Metric]) -> dict[str, ml.Metric]:
    return {metric.description: metric for metric in metrics}


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def save_metric_dataframes(metric_result: MetricResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_frame, plot_frame = metric_result.to_dataframe()
    if table_frame is not None:
        table_frame.to_csv(output_dir / "table_metrics.csv", index=False)
    if plot_frame is not None:
        plot_frame.to_csv(output_dir / "plot_metrics.csv", index=False)


# -----------------------------------------------------------------------------
# Dataset, network, and algorithms
# -----------------------------------------------------------------------------


def build_problem() -> tuple[benchmark.BenchmarkProblem, Any, list[str], int, int]:
    iop.set_seed(seed)
    train_dataset = FEMNISTDatasetHandler(
        split="train",
        n_clients=n_clients,
        train_fraction=train_fraction,
        seed=seed,
        min_train_samples=min_train_samples,
        min_test_samples=min_test_samples,
        local_files_only=local_files_only,
    )
    test_dataset = FEMNISTDatasetHandler(
        split="test",
        n_clients=n_clients,
        train_fraction=train_fraction,
        seed=seed,
        min_train_samples=min_train_samples,
        min_test_samples=min_test_samples,
        local_files_only=local_files_only,
    )

    train_partitions = train_dataset.get_partitions()
    costs = [
        PyTorchCost(
            dataset=partition,
            model=FEMNISTCNN(),
            loss_fn=torch.nn.CrossEntropyLoss(),
            final_activation=ArgmaxActivation(),
            batch_size=min(batch_size, len(partition)),
            device=device,
            load_dataset=load_dataset,
        )
        for partition in train_partitions
    ]
    agents = [
        Agent(
            cost,
            activation=AlwaysActive(),
            state_snapshot_period=state_snapshot_period,
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


def build_algorithm(
    algorithm_key: str,
    x0: Any,
    selected_hyperparameters: dict[str, Any],
    *,
    weighted_aggregation: bool,
) -> Any:
    aggregation_label = "data-size weighted" if weighted_aggregation else "uniform"

    if algorithm_key == "fedavg":
        params = algorithm_hyperparameters(selected_hyperparameters, "fedavg")
        return FedAvg(
            iterations=iterations,
            selection_scheme=make_selection_scheme(),
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedAvg {aggregation_label}",
            **params,
        )

    if algorithm_key == "fedprox":
        params = algorithm_hyperparameters(selected_hyperparameters, "fedprox")
        return FedProx(
            iterations=iterations,
            selection_scheme=make_selection_scheme(),
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedProx {aggregation_label}",
            **params,
        )

    if algorithm_key == "scaffold":
        params = algorithm_hyperparameters(selected_hyperparameters, "scaffold")
        return Scaffold(
            iterations=iterations,
            selection_scheme=make_selection_scheme(),
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"SCAFFOLD {aggregation_label}",
            **params,
        )

    if algorithm_key == "fednova":
        params = algorithm_hyperparameters(selected_hyperparameters, "fednova")
        params["num_local_steps"] = params.pop("num_local_epochs")
        return FedNova(
            iterations=iterations,
            selection_scheme=make_selection_scheme(),
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedNova {aggregation_label}",
            **params,
        )

    if algorithm_key == "fedopt":
        params = algorithm_hyperparameters(selected_hyperparameters, "fedopt")
        return FedAdam(
            iterations=iterations,
            selection_scheme=make_selection_scheme(),
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedAdam {aggregation_label}",
            **params,
        )

    if algorithm_key == "fedlt":
        params = algorithm_hyperparameters(selected_hyperparameters, "fedlt")
        return FedLT(
            iterations=iterations,
            selection_scheme=make_selection_scheme(),
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedLT {aggregation_label}",
            **params,
        )

    if algorithm_key == "feddyn":
        params = algorithm_hyperparameters(selected_hyperparameters, "feddyn")
        return FedDyn(
            iterations=iterations,
            selection_scheme=make_selection_scheme(),
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedDyn {aggregation_label}",
            **params,
        )

    raise ValueError(f"Unknown algorithm key: {algorithm_key}")


def build_algorithms(algorithm_key: str, x0: Any, selected_hyperparameters: dict[str, Any]) -> list[Any]:
    return [
        build_algorithm(algorithm_key, x0, selected_hyperparameters, weighted_aggregation=False),
        build_algorithm(algorithm_key, x0, selected_hyperparameters, weighted_aggregation=True),
    ]


# -----------------------------------------------------------------------------
# Metric merging
# -----------------------------------------------------------------------------


def merge_metric_result(
    source: MetricResult,
    *,
    combined_table_results: dict[Any, dict[ml.Metric, Any]],
    combined_plot_results: dict[Any, dict[ml.Metric, Any]],
    canonical_table_metrics: list[ml.Metric],
    canonical_plot_metrics: list[ml.Metric],
) -> None:
    table_lookup = metric_lookup(source.table_metrics or [])
    plot_lookup = metric_lookup(source.plot_metrics or [])
    canonical_table_lookup = metric_lookup(canonical_table_metrics)
    canonical_plot_lookup = metric_lookup(canonical_plot_metrics)

    for algorithm, metric_values in (source.table_results or {}).items():
        combined_table_results[algorithm] = {}
        for description, source_metric in table_lookup.items():
            if source_metric in metric_values:
                combined_table_results[algorithm][canonical_table_lookup[description]] = metric_values[source_metric]

    for algorithm, metric_values in (source.plot_results or {}).items():
        combined_plot_results[algorithm] = {}
        for description, source_metric in plot_lookup.items():
            if source_metric in metric_values:
                combined_plot_results[algorithm][canonical_plot_lookup[description]] = metric_values[source_metric]


# -----------------------------------------------------------------------------
# Metadata and execution
# -----------------------------------------------------------------------------


def write_run_inputs(
    *,
    run_path: Path,
    selected_writer_ids: list[str],
    n_train_samples: int,
    n_test_samples: int,
    requested_algorithms: list[str],
    statuses: list[dict[str, Any]],
) -> None:
    run_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": experiment_name,
        "title": "FEMNIST aggregation weighting benchmark",
        "purpose": "Compare uniform and data-size weighted aggregation for each tuned federated algorithm.",
        "execution": "one algorithm pair per benchmark() call; algorithms can be selected with --algorithm",
        "dataset": "FEMNIST",
        "dataset_source": "flwrlabs/femnist",
        "partition": "natural writer/client split",
        "n_clients": n_clients,
        "min_train_samples": min_train_samples,
        "min_test_samples": min_test_samples,
        "n_classes": 62,
        "train_fraction": train_fraction,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "n_trials": n_trials,
        "iterations": iterations,
        "state_snapshot_period": state_snapshot_period,
        "checkpoint_step": checkpoint_step,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "load_dataset": load_dataset,
        "selected_hyperparameters_path": str(selected_hyperparameters_path),
        "model": "CNN: conv 1->32, conv 32->64, dense 256, output 62 logits",
        "loss": "torch.nn.CrossEntropyLoss",
        "network": {
            "participation": "partial",
            "client_selection": "UniformSelection",
            "selection_fraction": selection_fraction,
            "clients_per_round": int(n_clients * selection_fraction),
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
        "excluded_algorithms": {
            "FedPD": (
                "Excluded from the main partial-participation FEMNIST experiments because the current implementation "
                "does not support client subsampling."
            )
        },
        "selected_writer_ids": selected_writer_ids,
        "statuses": statuses,
    }
    (run_path / "experiment2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_algorithm_pair(
    *,
    algorithm_key: str,
    selected_hyperparameters: dict[str, Any],
    run_path: Path,
    selected_writer_ids: list[str] | None,
    n_train_samples: int,
    n_test_samples: int,
    statuses: list[dict[str, Any]],
) -> tuple[MetricResult, list[str], int, int]:
    algorithm_path = run_path / "per_algorithm" / algorithm_key
    print(f"Running {algorithm_key}; results: {algorithm_path}")

    problem, x0, writer_ids, current_n_train_samples, current_n_test_samples = build_problem()
    if selected_writer_ids is None:
        selected_writer_ids = writer_ids
        n_train_samples = current_n_train_samples
        n_test_samples = current_n_test_samples
    elif writer_ids != selected_writer_ids:
        raise RuntimeError("Selected writer IDs changed between algorithm runs.")

    algorithms = build_algorithms(algorithm_key, x0, selected_hyperparameters)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=algorithm_path,
        checkpoint_step=checkpoint_step,
        keep_n_checkpoints=1,
        benchmark_metadata={
            "experiment": experiment_name,
            "algorithm": algorithm_key,
            "aggregation_variants": [algorithm.name for algorithm in algorithms],
            "n_trials": n_trials,
            "iterations": iterations,
            "state_snapshot_period": state_snapshot_period,
            "checkpoint_step": checkpoint_step,
        },
    )

    try:
        result = benchmark.benchmark(
            algorithms=algorithms,
            benchmark_problem=problem,
            n_trials=n_trials,
            max_processes=1,
            progress_step=progress_step,
            show_speed=True,
            show_trial=True,
            checkpoint_manager=checkpoint_manager,
            log_level=logging.INFO,
        )
        metric_result = benchmark.compute_metrics(
            benchmark_result=result,
            table_metrics=build_table_metrics(),
            plot_metrics=build_plot_metrics(),
            checkpoint_manager=checkpoint_manager,
            log_level=logging.INFO,
        )
        benchmark.display_metrics(
            metrics_result=metric_result,
            checkpoint_manager=checkpoint_manager,
            individual_plots=True,
            show_plots=show_plots,
            log_level=logging.INFO,
        )
        save_metric_dataframes(metric_result, checkpoint_manager.get_results_path())
        metric_result.agent_metrics = None
        save_pickle_zst(metric_result, algorithm_path / metric_result_filename)
        (algorithm_path / "metric_computation_complete.json").write_text(
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
        return metric_result, selected_writer_ids or [], n_train_samples, n_test_samples

    finally:
        if "metric_result" in locals():
            del metric_result
        if "result" in locals():
            del result
        del algorithms, problem, x0
        clear_cuda_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        default="all",
        choices=["all", *algorithm_order],
        help="Federated algorithm to run, or 'all' to run all experiment-2 algorithm pairs sequentially.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional suffix for the output run directory.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    requested_algorithms = list(algorithm_order) if args.algorithm == "all" else [args.algorithm]

    run_id = f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    if args.run_label:
        run_id = f"{run_id}_{slugify(args.run_label)}"
    checkpoint_root = Path("experiments/femnist/checkpoints") / experiment_group
    run_path = checkpoint_root / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    selected_hyperparameters = load_selected_hyperparameters()
    selected_writer_ids: list[str] | None = None
    n_train_samples = 0
    n_test_samples = 0
    statuses: list[dict[str, Any]] = []
    combined_table_results: dict[Any, dict[ml.Metric, Any]] = {}
    combined_plot_results: dict[Any, dict[ml.Metric, Any]] = {}
    canonical_table_metrics: list[ml.Metric] | None = None
    canonical_plot_metrics: list[ml.Metric] | None = None

    print(f"Writing Experiment 2 results to: {run_path}")

    try:
        for algorithm_key in requested_algorithms:
            metric_result, selected_writer_ids, n_train_samples, n_test_samples = run_algorithm_pair(
                algorithm_key=algorithm_key,
                selected_hyperparameters=selected_hyperparameters,
                run_path=run_path,
                selected_writer_ids=selected_writer_ids,
                n_train_samples=n_train_samples,
                n_test_samples=n_test_samples,
                statuses=statuses,
            )

            if canonical_table_metrics is None or canonical_plot_metrics is None:
                canonical_table_metrics = metric_result.table_metrics
                canonical_plot_metrics = metric_result.plot_metrics

            if canonical_table_metrics is None or canonical_plot_metrics is None:
                raise RuntimeError("Metric computation returned no table or plot metrics.")

            merge_metric_result(
                metric_result,
                combined_table_results=combined_table_results,
                combined_plot_results=combined_plot_results,
                canonical_table_metrics=canonical_table_metrics,
                canonical_plot_metrics=canonical_plot_metrics,
            )

    finally:
        write_run_inputs(
            run_path=run_path,
            selected_writer_ids=selected_writer_ids or [],
            n_train_samples=n_train_samples,
            n_test_samples=n_test_samples,
            requested_algorithms=requested_algorithms,
            statuses=statuses,
        )

    if canonical_table_metrics is None or canonical_plot_metrics is None:
        raise RuntimeError("No algorithm pair completed successfully.")

    combined_metric_result = MetricResult(
        agent_metrics=None,
        table_metrics=canonical_table_metrics,
        plot_metrics=canonical_plot_metrics,
        table_results=combined_table_results,
        plot_results=combined_plot_results,
    )
    combined_results_path = run_path / "results"
    benchmark.display_metrics(
        metrics_result=combined_metric_result,
        save_path=combined_results_path,
        individual_plots=True,
        show_plots=show_plots,
        log_level=logging.INFO,
    )
    save_metric_dataframes(combined_metric_result, combined_results_path)
    save_pickle_zst(combined_metric_result, run_path / metric_result_filename)
    (run_path / "metric_computation_complete.json").write_text(
        json.dumps({"metric_computation_complete": True}, indent=2),
        encoding="utf-8",
    )
    write_run_inputs(
        run_path=run_path,
        selected_writer_ids=selected_writer_ids or [],
        n_train_samples=n_train_samples,
        n_test_samples=n_test_samples,
        requested_algorithms=requested_algorithms,
        statuses=statuses,
    )

    print(f"Experiment 2 aggregation-weighting benchmark complete: {run_path}")


if __name__ == "__main__":
    main()
