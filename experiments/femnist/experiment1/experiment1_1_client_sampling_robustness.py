# ruff: noqa: ANN401, D103, DTZ005, E402, I001, INP001, T201

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
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
from decent_bench.costs import PyTorchCost
from decent_bench.metrics import metric_library as ml
from decent_bench.networks import FedNetwork
from decent_bench.schemes import AlwaysActive, NoCompression, NoDrops, NoNoise, UniformSelection
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.pytorch_utils import ArgmaxActivation
from decent_bench.utils.types import SupportedDevices

from experiments.femnist.src import FEMNISTCNN, FEMNISTDatasetHandler
from experiments.femnist.src.inspection_helpers import (
    FEMNIST_CLASS_NAMES,
    add_seeded_per_writer_train_test_split,
    choose_candidate_clients,
    client_stats,
    load_huggingface_metadata,
)


# -----------------------------------------------------------------------------
# Fixed FEMNIST benchmark setup
# -----------------------------------------------------------------------------

train_test_split_seed = 20260524
default_client_selection_seeds = (20260524, 20260525, 20260526, 20260537)
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
compute_metrics = True
show_plots = False

experiment_name = "experiment1_1_client_sampling_robustness"
output_root = Path("experiments/femnist/checkpoints") / experiment_name
selected_hyperparameters_path = Path("experiments/femnist/experiment0/selected_hyperparameters.json")
metric_result_filename = "metric_computation.pkl.zst"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FEMNIST clean-baseline robustness checks over different selected-client seeds."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(default_client_selection_seeds),
        help="Client-selection seeds to run.",
    )
    return parser.parse_args()


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


def build_table_metrics() -> list[ml.Metric]:
    return [
        ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
        ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
        ml.Loss([np.average]),
        ml.ClientDriftFromServer([min, np.average, max], x_log=False, y_log=False),
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


# -----------------------------------------------------------------------------
# Client selection plots and metadata
# -----------------------------------------------------------------------------


def select_clients_for_seed(client_selection_seed: int) -> pd.DataFrame:
    metadata = load_huggingface_metadata(Path("experiments/femnist/data/cache"), local_files_only=local_files_only)
    metadata = add_seeded_per_writer_train_test_split(
        metadata,
        train_fraction=train_fraction,
        seed=train_test_split_seed,
    )
    stats = client_stats(metadata)
    return choose_candidate_clients(
        stats,
        candidate_clients=n_clients,
        min_train_samples=min_train_samples,
        min_test_samples=min_test_samples,
        seed=client_selection_seed,
    )


def save_selected_client_artifacts(selected: pd.DataFrame, run_path: Path, *, client_selection_seed: int) -> None:
    selected.to_csv(run_path / "selected_clients_stats.csv", index=False)
    selected[["writer_id"]].to_csv(run_path / "selected_clients.csv", index=False)
    save_selected_client_class_distribution_plot(
        selected,
        run_path / "selected_client_class_distributions.png",
        title=f"Class distribution for selected writers, client-selection seed {client_selection_seed}",
    )


def save_selected_client_class_distribution_plot(selected: pd.DataFrame, path: Path, *, title: str) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    label_counts: np.ndarray = np.zeros((len(selected), len(FEMNIST_CLASS_NAMES)), dtype=np.int64)
    for row_index, histogram in enumerate(selected["label_histogram"]):
        parsed = cast("dict[str, int]", json.loads(histogram))
        for label, count in parsed.items():
            label_counts[row_index, int(label)] = int(count)

    figure_height = max(8, min(28, 2 + len(selected) // 4))
    fig, ax = plt.subplots(figsize=(18, figure_height))
    image = ax.imshow(label_counts, aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Writer ID")
    ax.set_xticks(np.arange(len(FEMNIST_CLASS_NAMES)))
    ax.set_xticklabels(FEMNIST_CLASS_NAMES, rotation=90)
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(selected["writer_id"].astype(str))
    fig.colorbar(image, ax=ax, label="Samples")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def selected_class_summary(selected: pd.DataFrame) -> dict[str, Any]:
    totals = dict.fromkeys(range(len(FEMNIST_CLASS_NAMES)), 0)
    for histogram in selected["label_histogram"]:
        parsed = cast("dict[str, int]", json.loads(histogram))
        for label, count in parsed.items():
            totals[int(label)] += int(count)
    missing = [label for label, count in totals.items() if count == 0]
    return {
        "selected_classes_covered": len(FEMNIST_CLASS_NAMES) - len(missing),
        "selected_missing_classes": missing,
    }


# -----------------------------------------------------------------------------
# Dataset, network, and algorithms
# -----------------------------------------------------------------------------


def build_problem(
    *,
    client_selection_seed: int,
    selected_clients_path: Path,
) -> tuple[benchmark.BenchmarkProblem, Any, list[str], int, int]:
    iop.set_seed(client_selection_seed)
    train_dataset = FEMNISTDatasetHandler(
        split="train",
        selected_clients_path=selected_clients_path,
        n_clients=n_clients,
        train_fraction=train_fraction,
        seed=train_test_split_seed,
        min_train_samples=min_train_samples,
        min_test_samples=min_test_samples,
        local_files_only=local_files_only,
    )
    test_dataset = FEMNISTDatasetHandler(
        split="test",
        selected_clients_path=selected_clients_path,
        n_clients=n_clients,
        train_fraction=train_fraction,
        seed=train_test_split_seed,
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


def build_algorithms(x0: Any, selected_hyperparameters: dict[str, Any]) -> list[Any]:
    algorithms: list[Any] = []

    fedavg_params = algorithm_hyperparameters(selected_hyperparameters, "fedavg")
    algorithms.append(FedAvg(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **fedavg_params))

    fedprox_params = algorithm_hyperparameters(selected_hyperparameters, "fedprox")
    algorithms.append(FedProx(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **fedprox_params))

    scaffold_params = algorithm_hyperparameters(selected_hyperparameters, "scaffold")
    algorithms.append(
        Scaffold(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **scaffold_params)
    )

    fednova_params = algorithm_hyperparameters(selected_hyperparameters, "fednova")
    fednova_params["num_local_steps"] = fednova_params.pop("num_local_epochs")
    algorithms.append(FedNova(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **fednova_params))

    fedopt_params = algorithm_hyperparameters(selected_hyperparameters, "fedopt")
    algorithms.append(FedAdam(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **fedopt_params))

    fedlt_params = algorithm_hyperparameters(selected_hyperparameters, "fedlt")
    algorithms.append(FedLT(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **fedlt_params))

    feddyn_params = algorithm_hyperparameters(selected_hyperparameters, "feddyn")
    algorithms.append(FedDyn(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **feddyn_params))

    return algorithms


# -----------------------------------------------------------------------------
# Metadata and execution
# -----------------------------------------------------------------------------


def build_run_metadata(
    *,
    client_selection_seed: int,
    selected: pd.DataFrame,
    selected_writer_ids: list[str],
    n_train_samples: int,
    n_test_samples: int,
    algorithms: list[Any],
    status: str,
) -> dict[str, Any]:
    class_summary = selected_class_summary(selected)
    return {
        "experiment": experiment_name,
        "purpose": "Robustness check for Experiment 1 conclusions under different selected FEMNIST clients.",
        "dataset": "FEMNIST",
        "dataset_source": "flwrlabs/femnist",
        "partition": "natural writer/client split",
        "n_clients": n_clients,
        "min_train_samples": min_train_samples,
        "min_test_samples": min_test_samples,
        "n_classes": 62,
        "train_fraction": train_fraction,
        "train_test_split_seed": train_test_split_seed,
        "client_selection_seed": client_selection_seed,
        "changed_variable": "client_selection_seed only",
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "n_trials": n_trials,
        "iterations": iterations,
        "state_snapshot_period": state_snapshot_period,
        "checkpoint_step": checkpoint_step,
        "batch_size": batch_size,
        "device": str(device),
        "local_files_only": local_files_only,
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
        "algorithms": [algorithm.name for algorithm in algorithms],
        "excluded_algorithms": {
            "FedPD": (
                "Excluded from the main partial-participation baseline because the current implementation "
                "does not support client subsampling."
            )
        },
        "selected_writer_ids": selected_writer_ids,
        "selected_total_train_samples": int(selected["train_samples"].sum()),
        "selected_total_test_samples": int(selected["test_samples"].sum()),
        "selected_min_train_samples": int(selected["train_samples"].min()),
        "selected_min_test_samples": int(selected["test_samples"].min()),
        "selected_median_train_samples": float(selected["train_samples"].median()),
        "selected_median_test_samples": float(selected["test_samples"].median()),
        **class_summary,
        "artifacts": {
            "selected_clients": "selected_clients.csv",
            "selected_client_stats": "selected_clients_stats.csv",
            "selected_client_class_distribution": "selected_client_class_distributions.png",
        },
        "status": status,
    }


def run_seed(client_selection_seed: int) -> None:
    run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_path = output_root / f"seed_{client_selection_seed}" / run_id
    staging_dir = output_root / "_selected_client_manifests"
    staging_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {experiment_name} seed {client_selection_seed} results to: {run_path}")

    selected = select_clients_for_seed(client_selection_seed)
    selected_clients_path = staging_dir / f"selected_clients_seed_{client_selection_seed}_{run_id}.csv"
    selected[["writer_id"]].to_csv(selected_clients_path, index=False)
    selected_hyperparameters = load_selected_hyperparameters()
    problem, x0, selected_writer_ids, n_train_samples, n_test_samples = build_problem(
        client_selection_seed=client_selection_seed,
        selected_clients_path=selected_clients_path,
    )
    algorithms = build_algorithms(x0, selected_hyperparameters)

    metadata = build_run_metadata(
        client_selection_seed=client_selection_seed,
        selected=selected,
        selected_writer_ids=selected_writer_ids,
        n_train_samples=n_train_samples,
        n_test_samples=n_test_samples,
        algorithms=algorithms,
        status="started",
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=run_path,
        checkpoint_step=checkpoint_step,
        keep_n_checkpoints=1,
        benchmark_metadata={
            "experiment": experiment_name,
            "client_selection_seed": client_selection_seed,
            "n_trials": n_trials,
            "iterations": iterations,
            "state_snapshot_period": state_snapshot_period,
            "checkpoint_step": checkpoint_step,
            "algorithms": [algorithm.name for algorithm in algorithms],
            "run_inputs": metadata,
        },
    )

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
    save_selected_client_artifacts(selected, run_path, client_selection_seed=client_selection_seed)
    (run_path / "experiment1_1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if compute_metrics:
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
        metric_result.agent_metrics = None
        save_pickle_zst(metric_result, run_path / metric_result_filename)
        (run_path / "metric_computation_complete.json").write_text(
            json.dumps({"metric_computation_complete": True}, indent=2),
            encoding="utf-8",
        )

    metadata["status"] = "complete"
    (run_path / "experiment1_1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    selected_clients_path.unlink(missing_ok=True)
    print(f"{experiment_name} seed {client_selection_seed} complete: {checkpoint_manager.checkpoint_dir}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for client_selection_seed in args.seeds:
        run_seed(client_selection_seed)


if __name__ == "__main__":
    main()
