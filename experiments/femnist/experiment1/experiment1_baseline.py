from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime
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
compute_metrics = True
show_plots = False

experiment_group = "experiment1"
experiment_name = "experiment1_baseline"
run_path = Path("experiments/femnist/checkpoints") / experiment_group / experiment_name / f"run_{datetime.now():%Y%m%d_%H%M%S}"
selected_hyperparameters_path = Path("experiments/femnist/experiment0/selected_hyperparameters.json")
metric_result_filename = "metric_computation.pkl.zst"


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


def build_algorithms(x0: Any, selected_hyperparameters: dict[str, Any]) -> list[Any]:
    algorithms: list[Any] = []

    fedavg_params = algorithm_hyperparameters(selected_hyperparameters, "fedavg")
    algorithms.append(FedAvg(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **fedavg_params))

    fedprox_params = algorithm_hyperparameters(selected_hyperparameters, "fedprox")
    algorithms.append(FedProx(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **fedprox_params))

    scaffold_params = algorithm_hyperparameters(selected_hyperparameters, "scaffold")
    algorithms.append(Scaffold(iterations=iterations, selection_scheme=make_selection_scheme(), x0=x0, **scaffold_params))

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

def write_run_inputs(
    *,
    selected_writer_ids: list[str],
    n_train_samples: int,
    n_test_samples: int,
    algorithms: list[Any],
) -> None:
    run_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": experiment_name,
        "purpose": "Baseline benchmark of tuned federated algorithms on the fixed FEMNIST setup.",
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
        "algorithms": [algorithm.name for algorithm in algorithms],
        "excluded_algorithms": {
            "FedPD": (
                "Excluded from the main partial-participation baseline because the current implementation "
                "does not support client subsampling."
            )
        },
        "selected_writer_ids": selected_writer_ids,
    }
    (run_path / "experiment1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    selected_hyperparameters = load_selected_hyperparameters()
    problem, x0, selected_writer_ids, n_train_samples, n_test_samples = build_problem()
    algorithms = build_algorithms(x0, selected_hyperparameters)
    write_run_inputs(
        selected_writer_ids=selected_writer_ids,
        n_train_samples=n_train_samples,
        n_test_samples=n_test_samples,
        algorithms=algorithms,
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=run_path,
        checkpoint_step=checkpoint_step,
        keep_n_checkpoints=1,
        benchmark_metadata={
            "experiment": experiment_name,
            "n_trials": n_trials,
            "iterations": iterations,
            "state_snapshot_period": state_snapshot_period,
            "checkpoint_step": checkpoint_step,
            "algorithms": [algorithm.name for algorithm in algorithms],
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

    if compute_metrics:
        metric_result = benchmark.compute_metrics(
            benchmark_result=result,
            table_metrics=[
                ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
                ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
                ml.Loss([np.average]),
                ml.ClientDriftFromServer([min, np.average, max], x_log=False, y_log=False),
                ml.GradientCalls([np.average, sum]),
                ml.SentMessages([np.average, sum]),
                ml.ReceivedMessages([np.average, sum]),
            ],
            plot_metrics=[
                ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
                ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
                ml.Loss([np.average]),
                ml.ClientDriftFromServer([], x_log=False, y_log=False),
            ],
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

    print(f"Experiment 1 baseline complete: {checkpoint_manager.checkpoint_dir}")


if __name__ == "__main__":
    main()
