# ruff: noqa: ANN401, D101, D103, E402, INP001, PLR0911, T201
"""
Fed-ISIC2019 communication-impairment robustness benchmark.

This experiment evaluates selected tuned federated algorithms under cross-silo
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
from collections.abc import Callable, Sequence
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
from decent_bench.algorithms.federated import (
    FedAdagrad,
    FedAdam,
    FedAvg,
    FedDyn,
    FedLT,
    FedNova,
    FedPD,
    FedProx,
    FedYogi,
    Scaffold,
)
from decent_bench.algorithms.utils import pytorch_initialization
from decent_bench.benchmark import MetricResult
from decent_bench.costs import PyTorchCost
from decent_bench.metrics._plots import _add_legend_and_save, _plot_subplot
from decent_bench.networks import FedNetwork
from decent_bench.schemes import (
    AgentActivationScheme,
    AlwaysActive,
    CompressionScheme,
    DropScheme,
    GaussianNoise,
    NoCompression,
    NoDrops,
    NoiseScheme,
    NoNoise,
    TopK,
    UniformActivationRate,
    UniformDropRate,
)
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.pytorch_utils import ArgmaxActivation
from decent_bench.utils.types import Datapoint, SupportedDevices
from experiments.fedisic2019.src import FedISICDatasetHandler, WeightedFocalLoss, build_model, class_weights_from_labels


seed = 20260524
n_trials = 3
iterations = 2500
state_snapshot_period = 250
progress_step = 200
checkpoint_step = None
batch_size = 64
device = SupportedDevices.GPU
local_files_only = False
load_dataset = False
show_plots = False
model_name = "efficientnet_b0"
pretrained = True
class_weight_mode = "flamby"

experiment_group = "experiment5"
experiment_name = "experiment5_communication_impairments"
selected_hyperparameters_path = Path("experiments/fedisic2019/selected_hyperparameters.json")
checkpoint_root = Path("experiments/fedisic2019/checkpoints") / experiment_group
metric_result_filename = "metric_computation.pkl.zst"

algorithm_order = (
    "fedavg",
    "fedprox",
    "scaffold",
    "fednova",
    "fedopt",
    "fedlt",
    "feddyn",
    "fedpd",
)


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
        impairment_label="AlwaysActive + full participation + no noise/compression/drops",
        activation_factory=AlwaysActive,
        compression_factory=NoCompression,
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={
            "activation": "AlwaysActive",
            "client_selection": None,
            "compression": "NoCompression",
            "drops": "NoDrops",
            "noise": "NoNoise",
        },
    ),
    Condition(
        key="availability",
        label="Availability",
        impairment_label="Availability: UniformActivationRate(0.80)",
        activation_factory=lambda: UniformActivationRate(0.80),
        compression_factory=NoCompression,
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={"activation": "UniformActivationRate", "activation_probability": 0.80},
    ),
    Condition(
        key="compression",
        label="Compression",
        impairment_label="Compression: TopK(0.10)",
        activation_factory=AlwaysActive,
        compression_factory=lambda: TopK(0.10),
        drop_factory=NoDrops,
        noise_factory=NoNoise,
        parameters={"compression": "TopK", "k": 0.10},
    ),
    Condition(
        key="drops",
        label="Drops",
        impairment_label="Drops: UniformDropRate(0.20)",
        activation_factory=AlwaysActive,
        compression_factory=NoCompression,
        drop_factory=lambda: UniformDropRate(0.20),
        noise_factory=NoNoise,
        parameters={"drops": "UniformDropRate", "drop_rate": 0.20},
    ),
    Condition(
        key="noise",
        label="Noise",
        impairment_label="Noise: GaussianNoise(mean=0.0, std=0.001)",
        activation_factory=AlwaysActive,
        compression_factory=NoCompression,
        drop_factory=NoDrops,
        noise_factory=lambda: GaussianNoise(mean=0.0, std=0.001),
        parameters={"noise": "GaussianNoise", "mean": 0.0, "std": 0.001},
    ),
    Condition(
        key="combination",
        label="Combination",
        impairment_label="UniformActivationRate(0.80) + TopK(0.10) + UniformDropRate(0.20) + GaussianNoise(0, 0.001)",
        activation_factory=lambda: UniformActivationRate(0.80),
        compression_factory=lambda: TopK(0.10),
        drop_factory=lambda: UniformDropRate(0.20),
        noise_factory=lambda: GaussianNoise(mean=0.0, std=0.001),
        parameters={
            "activation": "UniformActivationRate",
            "activation_probability": 0.80,
            "compression": "TopK",
            "k": 0.10,
            "drops": "UniformDropRate",
            "drop_rate": 0.20,
            "noise": "GaussianNoise",
            "mean": 0.0,
            "std": 0.001,
        },
    ),
)


def condition_lookup() -> dict[str, Condition]:
    return {condition.key: condition for condition in conditions}


def load_selected_hyperparameters() -> dict[str, Any]:
    with selected_hyperparameters_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _selected_algorithms(selected_hyperparameters: dict[str, Any]) -> dict[str, Any]:
    if "algorithms" in selected_hyperparameters:
        return dict(selected_hyperparameters["algorithms"])
    return selected_hyperparameters


def selected_entry(selected_hyperparameters: dict[str, Any], algorithm_key: str) -> dict[str, Any]:
    algorithms = _selected_algorithms(selected_hyperparameters)
    if algorithm_key in algorithms:
        entry = algorithms[algorithm_key]
    elif algorithm_key == "fedopt":
        # Backward-compatible fallback for the initial reference file before
        # the FedOpt family winner is collapsed under a single "fedopt" key.
        entry = next(
            algorithms[key]
            for key in ("fedadam", "fedyogi", "fedadagrad")
            if key in algorithms
        )
    else:
        raise KeyError(f"Missing selected hyperparameters for {algorithm_key!r}.")
    if "best_hyperparameters" in entry:
        return dict(entry["best_hyperparameters"])
    return dict(entry)


def selected_algorithm_name(selected_hyperparameters: dict[str, Any], algorithm_key: str) -> str:
    entry = selected_entry(selected_hyperparameters, algorithm_key)
    return str(entry.get("selected_algorithm") or entry.get("algorithm_name") or default_algorithm_name(algorithm_key))


def algorithm_hyperparameters(selected_hyperparameters: dict[str, Any], algorithm_key: str) -> dict[str, Any]:
    entry = selected_entry(selected_hyperparameters, algorithm_key)
    if "hyperparameters" not in entry:
        raise KeyError(f"Selected entry for {algorithm_key!r} does not include a 'hyperparameters' field.")
    return dict(entry["hyperparameters"])


def default_algorithm_name(algorithm_key: str) -> str:
    return {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "scaffold": "SCAFFOLD",
        "fednova": "FedNova",
        "fedopt": "FedAdam",
        "fedlt": "FedLT",
        "feddyn": "FedDyn",
        "fedpd": "FedPD",
    }[algorithm_key]


def labels_from_dataset(dataset: Sequence[Datapoint]) -> list[int]:
    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return [int(label) for label in labels]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def infer_num_classes(partitions: Sequence[Sequence[Datapoint]]) -> int:
    labels: list[int] = []
    for partition in partitions:
        labels.extend(labels_from_dataset(partition))
    if not labels:
        raise ValueError("Cannot infer number of classes from empty partitions.")
    return max(labels) + 1


def build_alpha(partitions: Sequence[Sequence[Datapoint]], num_classes: int) -> torch.Tensor | None:
    if class_weight_mode == "flamby":
        return None
    labels: list[int] = []
    for partition in partitions:
        labels.extend(labels_from_dataset(partition))
    return class_weights_from_labels(labels, num_classes)


def build_problem(condition: Condition) -> tuple[benchmark.BenchmarkProblem, Any, dict[str, Any]]:
    iop.set_seed(seed)
    train_dataset = FedISICDatasetHandler(split="train", local_files_only=local_files_only)
    test_dataset = FedISICDatasetHandler(split="test", local_files_only=local_files_only)

    train_partitions = train_dataset.get_partitions()
    num_classes = infer_num_classes(train_partitions)
    alpha = build_alpha(train_partitions, num_classes)

    costs = [
        PyTorchCost(
            dataset=partition,
            model=build_model(model_name, num_classes=num_classes, pretrained=pretrained),
            loss_fn=WeightedFocalLoss(alpha=alpha),
            final_activation=ArgmaxActivation(),
            batch_size=min(batch_size, len(partition)),
            max_batch_size=batch_size,
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
            data={"center_id": center_id, "center_name": train_dataset.center_names[center_id]},
        )
        for center_id, cost in zip(train_dataset.center_ids, costs, strict=True)
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
    metadata = {
        "center_ids": train_dataset.center_ids,
        "center_names": train_dataset.center_names,
        "class_names": train_dataset.class_names,
        "n_classes": train_dataset.n_targets,
        "n_train_samples": sum(len(partition) for partition in train_partitions),
        "n_test_samples": len(test_dataset.get_datapoints()),
    }
    return problem, x0, metadata


def build_algorithm(algorithm_key: str, x0: Any, selected_hyperparameters: dict[str, Any]) -> Any:
    params = algorithm_hyperparameters(selected_hyperparameters, algorithm_key)
    if algorithm_key == "fedavg":
        return FedAvg(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "fedprox":
        return FedProx(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "scaffold":
        return Scaffold(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "fednova":
        params["num_local_steps"] = params.pop("num_local_epochs")
        return FedNova(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "fedopt":
        algorithm_name = selected_algorithm_name(selected_hyperparameters, algorithm_key)
        fedopt_class = {"FedAdam": FedAdam, "FedYogi": FedYogi, "FedAdagrad": FedAdagrad}[algorithm_name]
        return fedopt_class(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "fedlt":
        return FedLT(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "feddyn":
        return FedDyn(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "fedpd":
        params["num_local_steps"] = params.pop("num_local_epochs")
        return FedPD(iterations=iterations, x0=x0, **params)
    raise ValueError(f"Unknown algorithm key: {algorithm_key}")


def build_algorithms(x0: Any, selected_hyperparameters: dict[str, Any], algorithm_keys: Sequence[str]) -> list[Any]:
    return [build_algorithm(algorithm_key, x0, selected_hyperparameters) for algorithm_key in algorithm_keys]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def save_annotated_metric_plots(metric_result: MetricResult, condition: Condition, output_dir: Path) -> None:
    if metric_result.plot_metrics is None or metric_result.plot_results is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metric_result.plot_metrics:
        fig, ax = plt.subplots(layout="constrained")
        has_data = False
        for algorithm_index, (algorithm, metric_values) in enumerate(metric_result.plot_results.items()):
            if metric not in metric_values:
                continue
            x, y_mean, y_min, y_max = metric_values[metric]
            _plot_subplot(ax, x, y_mean, y_min, y_max, algorithm.name, algorithm_index)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_title(f"Fed-ISIC2019 {condition.label}")
        ax.set_xlabel("iterations")
        ax.set_ylabel(metric.description)
        if metric.x_log:
            ax.set_xscale("log")
        if metric.y_log:
            ax.set_yscale("log")
        ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
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
        metric_slug = slugify(metric.description)
        _add_legend_and_save(fig, [ax], output_dir / f"{condition.key}_{metric_slug}.png")
        plt.close(fig)


def save_metric_dataframes(metric_result: MetricResult, output_dir: Path) -> None:
    table_frame, plot_frame = metric_result.to_dataframe()
    output_dir.mkdir(parents=True, exist_ok=True)
    if table_frame is not None:
        table_frame.to_csv(output_dir / "table_metrics.csv", index=False)
    if plot_frame is not None:
        plot_frame.to_csv(output_dir / "plot_metrics.csv", index=False)


def save_pickle_zst(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor(level=1)
    with path.open("wb") as file_obj, compressor.stream_writer(file_obj) as compressed_writer:
        pickle.dump(data, compressed_writer, protocol=pickle.HIGHEST_PROTOCOL)


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def write_metadata(
    *,
    run_path: Path,
    condition: Condition,
    requested_algorithms: Sequence[str],
    selected_hyperparameters: dict[str, Any],
    dataset_metadata: dict[str, Any],
    status: dict[str, Any],
) -> None:
    metadata = {
        "experiment": experiment_name,
        "title": "Fed-ISIC2019 communication impairment robustness benchmark",
        "purpose": (
            "Evaluate robustness of tuned Fed-ISIC2019 federated algorithms under cross-silo availability, "
            "compression, message-drop, additive-noise, and combined impairments."
        ),
        "execution": "one condition per launch; all configured algorithms are run together in one benchmark() call",
        "dataset": "Fed-ISIC2019",
        "dataset_source": "flwrlabs/fed-isic2019",
        "partition": "natural FLamby/Flower center split",
        "evaluation_split": "official Fed-ISIC2019 test split",
        "n_clients": len(dataset_metadata["center_ids"]),
        "n_classes": dataset_metadata["n_classes"],
        "class_names": dataset_metadata["class_names"],
        "center_ids": dataset_metadata["center_ids"],
        "center_names": dataset_metadata["center_names"],
        "n_train_samples": dataset_metadata["n_train_samples"],
        "n_test_samples": dataset_metadata["n_test_samples"],
        "n_trials": n_trials,
        "iterations": iterations,
        "state_snapshot_period": state_snapshot_period,
        "checkpoint_step": checkpoint_step,
        "batch_size": batch_size,
        "seed": seed,
        "device": device.value,
        "load_dataset": load_dataset,
        "selected_hyperparameters_path": str(selected_hyperparameters_path),
        "selected_hyperparameters": selected_hyperparameters,
        "model": model_name,
        "pretrained": pretrained,
        "loss": "WeightedFocalLoss(gamma=2.0, alpha=FLamby Fed-ISIC2019 weights by default)",
        "class_weight_mode": class_weight_mode,
        "network": {
            "participation": "full participation over active clients",
            "client_selection": None,
            "cross_silo_rationale": (
                "Cross-silo FL usually defaults to full participation because the number of institutions is small "
                "and coordination is more controlled than in cross-device FL."
            ),
            "availability_rationale": (
                "Cross-silo clients are generally assumed to be available and stable, but temporary unavailability "
                "can still happen due to maintenance windows, system failures, or connectivity incidents. This "
                "experiment therefore tests a high availability rate, UniformActivationRate(0.80)."
            ),
            "noise": "condition-specific",
            "compression": "condition-specific",
            "drops": "condition-specific",
            "impairments_apply_bidirectionally": True,
        },
        "condition": {
            "key": condition.key,
            "label": condition.label,
            "impairment_label": condition.impairment_label,
            "parameters": condition.parameters,
        },
        "condition_definitions": [
            {
                "key": configured_condition.key,
                "label": configured_condition.label,
                "impairment_label": configured_condition.impairment_label,
                "parameters": configured_condition.parameters,
            }
            for configured_condition in conditions
        ],
        "requested_algorithms": list(requested_algorithms),
        "status": status,
    }
    (run_path / "experiment5_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_condition(*, condition: Condition, run_path: Path, requested_algorithms: Sequence[str]) -> None:
    selected_hyperparameters = load_selected_hyperparameters()
    print(f"Running condition={condition.key}; results: {run_path}")
    problem, x0, dataset_metadata = build_problem(condition)
    algorithms = build_algorithms(x0, selected_hyperparameters, requested_algorithms)
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

    status: dict[str, Any] = {
        "condition": condition.key,
        "status": "started",
        "algorithms": [algorithm.name for algorithm in algorithms],
    }
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
            table_metrics=None,
            plot_metrics=None,
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
        save_metric_dataframes(metric_result, run_path / "metric_dataframes")
        save_annotated_metric_plots(metric_result, condition, run_path / "annotated_plots")
        metric_result.agent_metrics = None
        save_pickle_zst(metric_result, run_path / metric_result_filename)
        (run_path / "metric_computation_complete.json").write_text(
            json.dumps({"metric_computation_complete": True}, indent=2),
            encoding="utf-8",
        )
        status = {
            "condition": condition.key,
            "status": "ok",
            "algorithms": [algorithm.name for algorithm in algorithms],
        }
    except Exception as error:
        status = {
            "condition": condition.key,
            "status": "failed",
            "algorithms": [algorithm.name for algorithm in algorithms],
            "error": repr(error),
        }
        raise
    finally:
        write_metadata(
            run_path=run_path,
            condition=condition,
            requested_algorithms=requested_algorithms,
            selected_hyperparameters=selected_hyperparameters,
            dataset_metadata=dataset_metadata,
            status=status,
        )
        if "metric_result" in locals():
            del metric_result
        if "result" in locals():
            del result
        if "algorithms" in locals():
            del algorithms
        if "problem" in locals():
            del problem
        if "x0" in locals():
            del x0
        clear_cuda_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        default="clean_baseline",
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
    selected_condition = condition_lookup()[args.condition]
    run_id = f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    if args.run_label:
        run_id = f"{run_id}_{slugify(args.run_label)}"
    run_path = checkpoint_root / selected_condition.key / run_id

    run_condition(condition=selected_condition, run_path=run_path, requested_algorithms=algorithm_order)
    print(f"Experiment 5 communication-impairment benchmark complete: {run_path}")


if __name__ == "__main__":
    main()
