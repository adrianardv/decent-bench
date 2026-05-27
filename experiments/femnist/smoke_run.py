from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import decent_bench.utils.interoperability as iop
import torch
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
from decent_bench.costs import PyTorchCost
from decent_bench.networks import FedNetwork
from decent_bench.schemes import AlwaysActive, NoCompression, NoDrops, NoNoise
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.pytorch_utils import ArgmaxActivation
from decent_bench.utils.types import SupportedDevices

from experiments.femnist.src import FEMNISTCNN, FEMNISTDatasetHandler


checkpoint_path = Path("experiments/femnist/checkpoints/lambda_smoke_run") / f"run_{datetime.now():%Y%m%d_%H%M%S}"

seed = 20260524
n_clients = 100
min_train_samples = 100
min_test_samples = 20
train_fraction = 0.8
n_trials = 1
iterations = 1000
state_snapshot_period = 50
progress_step = iterations // 10
checkpoint_step = None
batch_size = 32
device = SupportedDevices.GPU
local_files_only = False
load_dataset = True
compute_metrics = True
show_plots = False

step_size = 0.01
num_local_epochs = 5
server_step_size = 0.001


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
    for partition in train_dataset.get_partitions()
]
agents = [
    Agent(
        cost,
        activation=AlwaysActive(),
        state_snapshot_period=state_snapshot_period,
        data={"writer_id": writer_id},
    )
    for writer_id, cost in zip(
        train_dataset.selected_writer_ids,
        costs,
        strict=True,
    )
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
algorithms = [
    FedAvg(
        iterations=iterations,
        step_size=step_size,
        num_local_epochs=num_local_epochs,
        selection_scheme=None,
        x0=x0,
    ),
    FedProx(
        iterations=iterations,
        step_size=step_size,
        num_local_epochs=num_local_epochs,
        mu=0.01,
        selection_scheme=None,
        x0=x0,
    ),
    Scaffold(
        iterations=iterations,
        step_size=step_size,
        num_local_epochs=num_local_epochs,
        server_step_size=1.0,
        selection_scheme=None,
        x0=x0,
    ),
    FedNova(
        iterations=iterations,
        step_size=step_size,
        num_local_steps=num_local_epochs,
        selection_scheme=None,
        x0=x0,
    ),
    FedAdam(
        iterations=iterations,
        step_size=step_size,
        num_local_epochs=num_local_epochs,
        server_step_size=server_step_size,
        selection_scheme=None,
        x0=x0,
    ),
    FedYogi(
        iterations=iterations,
        step_size=step_size,
        num_local_epochs=num_local_epochs,
        server_step_size=server_step_size,
        selection_scheme=None,
        x0=x0,
    ),
    FedAdagrad(
        iterations=iterations,
        step_size=step_size,
        num_local_epochs=num_local_epochs,
        server_step_size=server_step_size,
        selection_scheme=None,
        x0=x0,
    ),
    FedLT(
        iterations=iterations,
        step_size=step_size,
        num_local_epochs=num_local_epochs,
        rho=1.0,
        local_solver="gd",
        selection_scheme=None,
        x0=x0,
    ),
    FedDyn(
        iterations=iterations,
        step_size=step_size,
        num_local_epochs=num_local_epochs,
        alpha=0.01,
        selection_scheme=None,
        x0=x0,
    ),
    FedPD(
        iterations=iterations,
        step_size=step_size,
        num_local_steps=num_local_epochs,
        eta=1.0,
        skip_probability=0.0,
        x0=x0,
    ),
]

checkpoint_manager = CheckpointManager(
    checkpoint_dir=checkpoint_path,
    checkpoint_step=checkpoint_step,
    keep_n_checkpoints=1,
    benchmark_metadata={
        "experiment": "smoke_run",
        "dataset": "FEMNIST",
        "dataset_source": "flwrlabs/femnist",
        "partition": "natural writer/client split",
        "n_clients": n_clients,
        "min_train_samples": min_train_samples,
        "min_test_samples": min_test_samples,
        "train_fraction": train_fraction,
        "n_trials": n_trials,
        "iterations": iterations,
        "seed": seed,
        "batch_size": batch_size,
        "load_dataset": load_dataset,
        "model": "CNN: conv 1->32, conv 32->64, dense 256, output 62 logits",
        "loss": "torch.nn.CrossEntropyLoss",
        "network": {
            "participation": "full",
            "client_selection": None,
            "activation": "AlwaysActive",
            "drops": "NoDrops",
            "noise": "NoNoise",
            "compression": "NoCompression",
        },
        "algorithms": [algorithm.name for algorithm in algorithms],
        "shared_hyperparameters": {
            "step_size": step_size,
            "num_local_epochs": num_local_epochs,
            "server_step_size": server_step_size,
        },
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
        checkpoint_manager=checkpoint_manager,
    )
    benchmark.display_metrics(
        metrics_result=metric_result,
        checkpoint_manager=checkpoint_manager,
        show_plots=show_plots,
    )

print(f"Smoke run complete: {checkpoint_manager.checkpoint_dir}")
