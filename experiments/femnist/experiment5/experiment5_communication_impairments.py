# ruff: noqa: ANN401, D101, D103, E402, INP001, PLR0911, T201
"""
FEMNIST communication-impairment robustness benchmark.

The experiment evaluates how selected federated algorithms behave under client
availability, compression, message-drop, and additive-noise impairments.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
from decent_bench.schemes import (
    AgentActivationScheme,
    AlwaysActive,
    CompressionScheme,
    DropScheme,
    GaussianNoise,
    MarkovChainActivation,
    NoCompression,
    NoDrops,
    NoiseScheme,
    NoNoise,
    StochasticQuantization,
    TopK,
    UniformActivationRate,
    UniformDropRate,
    UniformSelection,
)
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

experiment_group = "experiment5"
experiment_name = "experiment5_communication_impairments"
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
# Condition definitions
# -----------------------------------------------------------------------------

ActivationFactory = Callable[[], AgentActivationScheme]
CompressionFactory = Callable[[], CompressionScheme]
DropFactory = Callable[[], DropScheme]
NoiseFactory = Callable[[], NoiseScheme]


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    impairment_label: str
    activation_factory: ActivationFactory
    compression_factory: CompressionFactory
    drop_factory: DropFactory
    noise_factory: NoiseFactory
    parameters: dict[str, Any]


conditions: tuple[Condition, ...] = (
    Condition(
        key="clean_baseline",
        label="Clean baseline",
        impairment_label="No impairments",
        activation_factory=AlwaysActive,
        compression_factory=NoCompression,
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={"activation": "AlwaysActive", "compression": "NoCompression", "drops": "NoDrops"},
    ),
    Condition(
        key="activation_uniform_low",
        label="Uniform activation, low rate",
        impairment_label="Availability: Uniform p=0.30",
        activation_factory=lambda: UniformActivationRate(0.30),
        compression_factory=NoCompression,
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={"activation": "UniformActivationRate", "activation_probability": 0.30},
    ),
    Condition(
        key="activation_uniform_high",
        label="Uniform activation, high rate",
        impairment_label="Availability: Uniform p=0.80",
        activation_factory=lambda: UniformActivationRate(0.80),
        compression_factory=NoCompression,
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={"activation": "UniformActivationRate", "activation_probability": 0.80},
    ),
    Condition(
        key="activation_markov_high_availability",
        label="Markov activation, high availability",
        impairment_label="Availability: Markov 0.20/0.10",
        activation_factory=lambda: MarkovChainActivation(inactive_to_active=0.20, active_to_inactive=0.10),
        compression_factory=NoCompression,
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={"activation": "MarkovChainActivation", "inactive_to_active": 0.20, "active_to_inactive": 0.10},
    ),
    Condition(
        key="activation_markov_low_availability",
        label="Markov activation, low availability",
        impairment_label="Availability: Markov 0.10/0.30",
        activation_factory=lambda: MarkovChainActivation(inactive_to_active=0.10, active_to_inactive=0.30),
        compression_factory=NoCompression,
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={"activation": "MarkovChainActivation", "inactive_to_active": 0.10, "active_to_inactive": 0.30},
    ),
    Condition(
        key="compression_topk_low",
        label="Top-k compression, low k",
        impairment_label="Compression: Top-k 1%",
        activation_factory=AlwaysActive,
        compression_factory=lambda: TopK(0.01),
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={"compression": "TopK", "k": 0.01},
    ),
    Condition(
        key="compression_topk_high",
        label="Top-k compression, high k",
        impairment_label="Compression: Top-k 10%",
        activation_factory=AlwaysActive,
        compression_factory=lambda: TopK(0.10),
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={"compression": "TopK", "k": 0.10},
    ),
    Condition(
        key="compression_qsgd_low",
        label="Stochastic quantization, low levels",
        impairment_label="Compression: QSGD s=4",
        activation_factory=AlwaysActive,
        compression_factory=lambda: StochasticQuantization(4),
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={"compression": "StochasticQuantization", "n_levels": 4},
    ),
    Condition(
        key="compression_qsgd_high",
        label="Stochastic quantization, high levels",
        impairment_label="Compression: QSGD s=16",
        activation_factory=AlwaysActive,
        compression_factory=lambda: StochasticQuantization(16),
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={"compression": "StochasticQuantization", "n_levels": 16},
    ),
    Condition(
        key="drops_uniform_low",
        label="Uniform message drops, low rate",
        impairment_label="Drops: Uniform p=0.05",
        activation_factory=AlwaysActive,
        compression_factory=NoCompression,
        drop_factory=lambda: UniformDropRate(0.05),
        noise_factory=NoNoise,
        parameters={"drops": "UniformDropRate", "drop_rate": 0.05},
    ),
    Condition(
        key="drops_uniform_high",
        label="Uniform message drops, high rate",
        impairment_label="Drops: Uniform p=0.50",
        activation_factory=AlwaysActive,
        compression_factory=NoCompression,
        drop_factory=lambda: UniformDropRate(0.50),
        noise_factory=NoNoise,
        parameters={"drops": "UniformDropRate", "drop_rate": 0.50},
    ),
    Condition(
        key="noise_gaussian_low",
        label="Gaussian noise, low",
        impairment_label="Noise: Gaussian mean=0, std=0.001",
        activation_factory=AlwaysActive,
        compression_factory=NoCompression,
        drop_factory=NoDrops,
        noise_factory=lambda: GaussianNoise(mean=0.0, std=0.001),
        parameters={"noise": "GaussianNoise", "mean": 0.0, "std": 0.001},
    ),
    Condition(
        key="noise_gaussian_high",
        label="Gaussian noise, high",
        impairment_label="Noise: Gaussian mean=0, std=0.01",
        activation_factory=AlwaysActive,
        compression_factory=NoCompression,
        drop_factory=NoDrops,
        noise_factory=lambda: GaussianNoise(mean=0.0, std=0.01),
        parameters={"noise": "GaussianNoise", "mean": 0.0, "std": 0.01},
    ),
    Condition(
        key="combined_uniform_topk_drops",
        label="Combined impairments",
        impairment_label="Combination of impairments",
        activation_factory=lambda: UniformActivationRate(0.50),
        compression_factory=lambda: TopK(0.10),
        drop_factory=lambda: UniformDropRate(0.10),
        noise_factory=lambda: GaussianNoise(mean=0.0, std=0.001),
        parameters={
            "activation": "UniformActivationRate",
            "activation_probability": 0.50,
            "compression": "TopK",
            "k": 0.10,
            "drops": "UniformDropRate",
            "drop_rate": 0.10,
            "noise": "GaussianNoise",
            "mean": 0.0,
            "std": 0.001,
        },
    ),
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def condition_lookup() -> dict[str, Condition]:
    return {condition.key: condition for condition in conditions}


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
        ml.SentMessagesDropped([np.average, sum]),
    ]


def build_plot_metrics() -> list[ml.Metric]:
    return [
        ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
        ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
        ml.Loss([np.average]),
        ml.ClientDriftFromServer([], x_log=False, y_log=False),
    ]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


# -----------------------------------------------------------------------------
# Dataset, network, and algorithms
# -----------------------------------------------------------------------------


def build_problem(condition: Condition) -> tuple[benchmark.BenchmarkProblem, Any, list[str], int, int]:
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
            activation=condition.activation_factory(),
            state_snapshot_period=state_snapshot_period,
            data={"writer_id": writer_id},
        )
        for writer_id, cost in zip(train_dataset.selected_writer_ids, costs, strict=True)
    ]
    network = FedNetwork(
        clients=agents,
        message_noise=condition.noise_factory(),
        message_compression=condition.compression_factory(),
        message_drop=condition.drop_factory(),
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


def build_algorithm(algorithm_key: str, x0: Any, selected_hyperparameters: dict[str, Any]) -> Any:
    if algorithm_key == "fedavg":
        params = algorithm_hyperparameters(selected_hyperparameters, "fedavg")
        return FedAvg(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **params)

    if algorithm_key == "fedprox":
        params = algorithm_hyperparameters(selected_hyperparameters, "fedprox")
        return FedProx(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **params)

    if algorithm_key == "scaffold":
        params = algorithm_hyperparameters(selected_hyperparameters, "scaffold")
        return Scaffold(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **params)

    if algorithm_key == "fednova":
        params = algorithm_hyperparameters(selected_hyperparameters, "fednova")
        params["num_local_steps"] = params.pop("num_local_epochs")
        return FedNova(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **params)

    if algorithm_key == "fedopt":
        params = algorithm_hyperparameters(selected_hyperparameters, "fedopt")
        return FedAdam(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **params)

    if algorithm_key == "fedlt":
        params = algorithm_hyperparameters(selected_hyperparameters, "fedlt")
        return FedLT(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **params)

    if algorithm_key == "feddyn":
        params = algorithm_hyperparameters(selected_hyperparameters, "feddyn")
        return FedDyn(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **params)

    raise ValueError(f"Unknown algorithm key: {algorithm_key}")


def build_algorithms(x0: Any, selected_hyperparameters: dict[str, Any], algorithm_keys: list[str]) -> list[Any]:
    return [build_algorithm(algorithm_key, x0, selected_hyperparameters) for algorithm_key in algorithm_keys]


# -----------------------------------------------------------------------------
# Annotated plots
# -----------------------------------------------------------------------------


def save_annotated_metric_plots(metric_result: MetricResult, condition: Condition, output_dir: Path) -> None:
    if metric_result.plot_metrics is None or metric_result.plot_results is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metric_result.plot_metrics:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        has_data = False
        for algorithm, metric_values in metric_result.plot_results.items():
            if metric not in metric_values:
                continue
            x, y_mean, y_min, y_max = metric_values[metric]
            ax.plot(x, y_mean, marker="o", markersize=3, linewidth=1.5, label=algorithm.name)
            ax.fill_between(x, y_min, y_max, alpha=0.10)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("iterations")
        ax.set_ylabel(metric.description)
        ax.grid(visible=True, alpha=0.25)
        ax.text(
            0.98,
            0.08,
            condition.impairment_label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "square,pad=0.25", "facecolor": "#eeeeee", "edgecolor": "#777777", "alpha": 0.95},
        )
        ax.legend(loc="best", fontsize=8)
        metric_slug = slugify(metric.description)
        fig.tight_layout()
        fig.savefig(output_dir / f"{condition.key}_{metric_slug}.png", dpi=600)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Metadata and execution
# -----------------------------------------------------------------------------


def write_run_inputs(
    *,
    run_path: Path,
    selected_writer_ids: list[str],
    n_train_samples: int,
    n_test_samples: int,
    requested_conditions: list[str],
    requested_algorithms: list[str],
    statuses: list[dict[str, Any]],
) -> None:
    run_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": experiment_name,
        "title": "FEMNIST communication impairment robustness benchmark",
        "purpose": (
            "Evaluate the robustness of selected tuned federated algorithms under client availability, "
            "message compression, message-drop, and additive-noise impairments."
        ),
        "execution": "one planned condition per launch; all algorithms are run together in one benchmark() call",
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
            "clients_per_round_without_activation_impairment": int(n_clients * selection_fraction),
            "noise": "condition-specific",
            "impairments_apply_bidirectionally": True,
        },
        "requested_conditions": requested_conditions,
        "condition_definitions": [
            {
                "key": condition.key,
                "label": condition.label,
                "impairment_label": condition.impairment_label,
                "parameters": condition.parameters,
            }
            for condition in conditions
        ],
        "requested_algorithms": requested_algorithms,
        "algorithms": list(algorithm_order),
        "selected_writer_ids": selected_writer_ids,
        "statuses": statuses,
    }
    (run_path / "experiment5_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_condition(
    *,
    condition: Condition,
    algorithm_order: list[str],
    selected_hyperparameters: dict[str, Any],
    run_path: Path,
    selected_writer_ids: list[str] | None,
    n_train_samples: int,
    n_test_samples: int,
    statuses: list[dict[str, Any]],
) -> tuple[list[str], int, int]:
    print(f"Running condition={condition.key}; results: {run_path}")
    problem, x0, writer_ids, current_n_train_samples, current_n_test_samples = build_problem(condition)
    if selected_writer_ids is None:
        selected_writer_ids = writer_ids
        n_train_samples = current_n_train_samples
        n_test_samples = current_n_test_samples
    elif writer_ids != selected_writer_ids:
        raise RuntimeError("Selected writer IDs changed between runs.")

    algorithms = build_algorithms(x0, selected_hyperparameters, algorithm_order)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=run_path,
        checkpoint_step=checkpoint_step,
        keep_n_checkpoints=1,
        benchmark_metadata={
            "experiment": experiment_name,
            "condition": condition.key,
            "n_trials": n_trials,
            "iterations": iterations,
            "state_snapshot_period": state_snapshot_period,
            "checkpoint_step": checkpoint_step,
            "algorithms": [algorithm.name for algorithm in algorithms],
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
        save_annotated_metric_plots(metric_result, condition, run_path / "annotated_plots")
        metric_result.agent_metrics = None
        save_pickle_zst(metric_result, run_path / metric_result_filename)
        (run_path / "metric_computation_complete.json").write_text(
            json.dumps({"metric_computation_complete": True}, indent=2),
            encoding="utf-8",
        )
        statuses.append(
            {
                "condition": condition.key,
                "status": "ok",
                "algorithms": [algorithm.name for algorithm in algorithms],
            }
        )

    except Exception as error:
        statuses.append(
            {
                "condition": condition.key,
                "status": "failed",
                "algorithms": [algorithm.name for algorithm in algorithms],
                "error": repr(error),
            }
        )
        raise

    else:
        return selected_writer_ids or [], n_train_samples, n_test_samples

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
        "--conditions",
        nargs=1,
        default=["clean_baseline"],
        choices=[condition.key for condition in conditions],
        help="One planned condition key to run.",
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
    requested_conditions = args.conditions
    unknown_conditions = sorted(set(requested_conditions) - set(condition_lookup()))
    if unknown_conditions:
        choices = ", ".join(condition.key for condition in conditions)
        raise ValueError(f"Unknown condition(s): {unknown_conditions}. Valid choices: {choices}")
    requested_algorithms = list(algorithm_order)
    if len(requested_conditions) != 1:
        raise ValueError("Run exactly one planned condition at a time.")
    selected_condition = condition_lookup()[requested_conditions[0]]
    run_id = f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    if args.run_label:
        run_id = f"{run_id}_{slugify(args.run_label)}"
    checkpoint_root = Path("experiments/femnist/checkpoints") / experiment_group
    run_path = checkpoint_root / selected_condition.key / run_id

    selected_hyperparameters = load_selected_hyperparameters()
    selected_writer_ids: list[str] | None = None
    n_train_samples = 0
    n_test_samples = 0
    statuses: list[dict[str, Any]] = []

    print(f"Writing Experiment 5 results to: {run_path}")

    try:
        selected_writer_ids, n_train_samples, n_test_samples = run_condition(
            condition=selected_condition,
            algorithm_order=requested_algorithms,
            selected_hyperparameters=selected_hyperparameters,
            run_path=run_path,
            selected_writer_ids=selected_writer_ids,
            n_train_samples=n_train_samples,
            n_test_samples=n_test_samples,
            statuses=statuses,
        )
    finally:
        write_run_inputs(
            run_path=run_path,
            selected_writer_ids=selected_writer_ids or [],
            n_train_samples=n_train_samples,
            n_test_samples=n_test_samples,
            requested_conditions=requested_conditions,
            requested_algorithms=requested_algorithms,
            statuses=statuses,
        )

    print(f"Experiment 5 communication-impairment benchmark complete: {run_path}")


if __name__ == "__main__":
    main()
