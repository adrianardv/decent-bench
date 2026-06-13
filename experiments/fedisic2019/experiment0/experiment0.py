from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import pickle
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import decent_bench.utils.interoperability as iop
import numpy as np
import torch
import zstandard as zstd
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
from decent_bench.metrics import metric_library as ml
from decent_bench.networks import FedNetwork
from decent_bench.schemes import AlwaysActive, NoCompression, NoDrops, NoNoise
from decent_bench.utils.pytorch_utils import ArgmaxActivation
from decent_bench.utils.types import Datapoint, SupportedDevices

from experiments.fedisic2019.src import FedISICDatasetHandler, WeightedFocalLoss, build_model, class_weights_from_labels


seed = 20260524
validation_fraction = 0.2
default_batch_size = 64
default_debug_batch_size = 32
metric_result_filename = "metric_computation.pkl.zst"
selection_metric = "server balanced accuracy"
tie_break_metric = "loss"
algorithm_choices = [
    "fedavg",
    "fedprox",
    "scaffold",
    "fednova",
    "fedopt",
    "fedlt",
    "feddyn",
    "fedpd",
]
local_epoch_choices = [1, 2, 3, 4, 5]
fednova_mu_choices = [0.0005, 0.001, 0.005, 0.01]

candidate_result_fields = [
    "status",
    "search_stage",
    "candidate_id",
    "algorithm_key",
    "algorithm_group",
    "algorithm_name",
    "variant",
    "server_balanced_accuracy_mean",
    "server_balanced_accuracy_margin_of_error",
    "server_accuracy_mean",
    "validation_loss_mean",
    "validation_loss_margin_of_error",
    "elapsed_seconds",
    "error",
    "step_size",
    "num_local_epochs",
    "mu",
    "server_step_size",
    "beta",
    "beta_1",
    "beta_2",
    "tau",
    "use_momentum",
    "use_prox",
    "use_server_momentum",
    "gamma",
    "rho",
    "local_solver",
    "solver_args.beta1",
    "solver_args.beta2",
    "solver_args.epsilon",
    "solver_args.momentum",
    "alpha",
    "eta",
    "skip_probability",
]


@dataclass(frozen=True)
class RuntimeConfig:
    algorithm: str
    iterations: int
    final_iterations: int
    n_trials: int
    n_random_candidates: int
    max_grid_candidates: int
    run_final: bool
    run_path: Path
    batch_size: int
    device: SupportedDevices
    model_name: str
    pretrained: bool
    class_weight_mode: str
    max_samples_per_client: int | None
    local_files_only: bool
    load_dataset: bool


@dataclass(frozen=True)
class Candidate:
    algorithm_key: str
    group: str
    algorithm_name: str
    variant: str
    search_stage: str
    hyperparameters: dict[str, Any]

    @property
    def candidate_id(self) -> str:
        parts = [self.search_stage, self.group, self.algorithm_name, self.variant]
        compact = "_".join(part for part in parts if part)
        return "".join(char if char.isalnum() or char in "-_" else "_" for char in compact)


class IndexedSubset(Sequence[Datapoint]):
    """Lazy subset wrapper that keeps Fed-ISIC image tensors out of memory."""

    def __init__(self, dataset: Sequence[Datapoint], indices: Sequence[int], labels: Sequence[int] | None = None) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.labels = list(labels) if labels is not None else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Datapoint:
        return self.dataset[self.indices[index]]


class ConcatDataset(Sequence[Datapoint]):
    """Lazy concatenation for validation/test datasets."""

    def __init__(self, datasets: Sequence[Sequence[Datapoint]]) -> None:
        self.datasets = list(datasets)
        self._cumulative_lengths = np.cumsum([len(dataset) for dataset in self.datasets]).tolist()

    def __len__(self) -> int:
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0

    def __getitem__(self, index: int) -> Datapoint:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = int(np.searchsorted(self._cumulative_lengths, index, side="right"))
        previous_length = 0 if dataset_index == 0 else self._cumulative_lengths[dataset_index - 1]
        return self.datasets[dataset_index][index - previous_length]


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Tune Fed-ISIC2019 hyperparameters for one federated algorithm.")
    parser.add_argument("--algorithm", choices=algorithm_choices, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--final-iterations", type=int, default=1500)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--n-random-candidates", type=int, default=8)
    parser.add_argument("--max-grid-candidates", type=int, default=18)
    parser.add_argument("--skip-final-run", action="store_true")
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--device", choices=[device.value for device in SupportedDevices], default=SupportedDevices.GPU.value)
    parser.add_argument("--model", choices=["efficientnet_b0", "small_cnn"], default="efficientnet_b0")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--class-weight-mode", choices=["flamby", "computed"], default="flamby")
    parser.add_argument("--max-samples-per-client", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--load-dataset", action="store_true", help="Materialize lazy image datasets inside PyTorchCost.")
    args = parser.parse_args()

    run_name = args.run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_path = Path("experiments/fedisic2019/checkpoints/experiment0") / args.algorithm / run_name
    batch_size = args.batch_size
    if args.model == "small_cnn" and args.batch_size == default_batch_size:
        batch_size = default_debug_batch_size
    return RuntimeConfig(
        algorithm=args.algorithm,
        iterations=args.iterations,
        final_iterations=args.final_iterations,
        n_trials=args.n_trials,
        n_random_candidates=args.n_random_candidates,
        max_grid_candidates=args.max_grid_candidates,
        run_final=not args.skip_final_run,
        run_path=run_path,
        batch_size=batch_size,
        device=SupportedDevices(args.device),
        model_name=args.model,
        pretrained=not args.no_pretrained,
        class_weight_mode=args.class_weight_mode,
        max_samples_per_client=args.max_samples_per_client,
        local_files_only=args.local_files_only,
        load_dataset=args.load_dataset,
    )


def candidate(
    algorithm_key: str,
    group: str,
    algorithm_name: str,
    variant: str,
    search_stage: str,
    hyperparameters: dict[str, Any],
) -> Candidate:
    return Candidate(
        algorithm_key=algorithm_key,
        group=group,
        algorithm_name=algorithm_name,
        variant=variant,
        search_stage=search_stage,
        hyperparameters=hyperparameters,
    )


def build_algorithm(tuning_candidate: Candidate, x0: Any, iterations: int) -> Any:
    params = dict(tuning_candidate.hyperparameters)
    algorithm_key = tuning_candidate.algorithm_key
    algorithm_name = tuning_candidate.algorithm_name

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
        fedopt_class = {"FedAdam": FedAdam, "FedYogi": FedYogi, "FedAdagrad": FedAdagrad}[algorithm_name]
        return fedopt_class(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "fedlt":
        return FedLT(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "feddyn":
        return FedDyn(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "fedpd":
        params["num_local_steps"] = params.pop("num_local_epochs")
        return FedPD(iterations=iterations, x0=x0, **params)
    raise ValueError(f"Unsupported algorithm key: {algorithm_key}")


def reference_candidates(algorithm_key: str) -> list[Candidate]:
    refs = {
        "fedavg": [
            {"step_size": 0.01, "num_local_epochs": 1},
            {"step_size": 0.01, "num_local_epochs": 2},
            {"step_size": 0.01, "num_local_epochs": 4},
        ],
        "fedprox": [
            {"step_size": 0.01, "num_local_epochs": 1, "mu": 0.001},
            {"step_size": 0.01, "num_local_epochs": 2, "mu": 0.001},
            {"step_size": 0.01, "num_local_epochs": 4, "mu": 0.001},
        ],
        "scaffold": [
            {"step_size": 0.01, "num_local_epochs": 1, "server_step_size": 1.0},
            {"step_size": 0.01, "num_local_epochs": 2, "server_step_size": 1.0},
            {"step_size": 0.01, "num_local_epochs": 5, "server_step_size": 1.0},
        ],
        "fednova": [
            {
                "step_size": 0.01,
                "num_local_epochs": 3,
                "use_momentum": True,
                "use_server_momentum": False,
                "use_prox": False,
                "beta": 0.5,
            },
            {
                "step_size": 0.01,
                "num_local_epochs": 3,
                "use_momentum": False,
                "use_server_momentum": True,
                "use_prox": False,
                "gamma": 0.9,
            },
            {
                "step_size": 0.01,
                "num_local_epochs": 3,
                "use_momentum": False,
                "use_server_momentum": False,
                "use_prox": True,
                "mu": 0.001,
            },
            {
                "step_size": 0.015780201353739066,
                "num_local_epochs": 3,
                "use_momentum": True,
                "use_server_momentum": True,
                "use_prox": False,
                "beta": 0.5,
                "gamma": 0.9,
            },
            {
                "step_size": 0.01,
                "num_local_epochs": 3,
                "use_momentum": True,
                "use_server_momentum": True,
                "use_prox": True,
                "beta": 0.5,
                "gamma": 0.9,
                "mu": 0.001,
            },
        ],
        "fedopt": [
            {
                "algorithm_name": "FedAdam",
                "step_size": 0.01,
                "num_local_epochs": 2,
                "server_step_size": 0.0031622777,
                "beta_1": 0.9,
                "beta_2": 0.9,
                "tau": 0.001,
            },
            {
                "algorithm_name": "FedYogi",
                "step_size": 0.01,
                "num_local_epochs": 2,
                "server_step_size": 0.0031622777,
                "beta_1": 0.9,
                "beta_2": 0.9,
                "tau": 0.001,
            },
            {
                "algorithm_name": "FedAdagrad",
                "step_size": 0.01,
                "num_local_epochs": 2,
                "server_step_size": 0.0316227766,
                "beta_1": 0.0,
                "tau": 0.001,
            },
        ],
        "fedlt": [
            {
                "step_size": 0.0015,
                "num_local_epochs": 5,
                "rho": 1.0,
                "local_solver": "gd",
            },
            {
                "step_size": 0.0015,
                "num_local_epochs": 5,
                "rho": 1.0,
                "local_solver": "nesterov",
                "solver_args": {"momentum": 0.9},
            },
            {
                "step_size": 0.0015,
                "num_local_epochs": 5,
                "rho": 1.0,
                "local_solver": "adam",
                "solver_args": {"beta1": 0.5, "beta2": 0.999, "epsilon": 1e-8},
            }
        ],
        "feddyn": [{"step_size": 0.02760842017693185, "num_local_epochs": 2, "alpha": 1.0}],
        "fedpd": [{"step_size": 0.03, "num_local_epochs": 5, "eta": 0.3, "skip_probability": 0.2}],
    }
    candidates = []
    for index, params in enumerate(refs[algorithm_key]):
        params = dict(params)
        algorithm_name = params.pop("algorithm_name", _default_algorithm_name(algorithm_key))
        candidates.append(candidate(algorithm_key, _group_name(algorithm_key), algorithm_name, f"reference_{index}", "reference", params))
    return candidates


def random_candidates(algorithm_key: str, rng: np.random.Generator, n_candidates: int) -> list[Candidate]:
    builders = {
        "fedavg": random_fedavg_candidate,
        "fedprox": random_fedprox_candidate,
        "scaffold": random_scaffold_candidate,
        "fednova": random_fednova_candidate,
        "fedopt": random_fedopt_candidate,
        "fedlt": random_fedlt_candidate,
        "feddyn": random_feddyn_candidate,
        "fedpd": random_fedpd_candidate,
    }
    return [builders[algorithm_key](rng, index) for index in range(n_candidates)]


def random_fedavg_candidate(rng: np.random.Generator, index: int) -> Candidate:
    return candidate(
        "fedavg",
        "FedAvg",
        "FedAvg",
        f"random_{index:02d}",
        "random",
        {"step_size": log_uniform(rng, 1e-4, 5e-2), "num_local_epochs": random_choice(rng, local_epoch_choices)},
    )


def random_fedprox_candidate(rng: np.random.Generator, index: int) -> Candidate:
    return candidate(
        "fedprox",
        "FedProx",
        "FedProx",
        f"random_{index:02d}",
        "random",
        {
            "step_size": log_uniform(rng, 1e-4, 5e-2),
            "num_local_epochs": random_choice(rng, local_epoch_choices),
            "mu": log_uniform(rng, 1e-4, 1e-1),
        },
    )


def random_scaffold_candidate(rng: np.random.Generator, index: int) -> Candidate:
    return candidate(
        "scaffold",
        "SCAFFOLD",
        "Scaffold",
        f"random_{index:02d}",
        "random",
        {
            "step_size": log_uniform(rng, 1e-4, 5e-2),
            "num_local_epochs": random_choice(rng, local_epoch_choices),
            "server_step_size": random_choice(rng, [0.5, 1.0]),
        },
    )


def random_fednova_candidate(rng: np.random.Generator, index: int) -> Candidate:
    variant = random_choice(rng, ["momentum", "prox", "server_momentum", "both_momentums", "all_three"])
    params: dict[str, Any] = {
        "step_size": log_uniform(rng, 1e-4, 5e-2),
        "num_local_epochs": random_choice(rng, local_epoch_choices),
    }
    if variant in {"momentum", "both_momentums", "all_three"}:
        params.update({"use_momentum": True, "beta": random_choice(rng, [0.5, 0.9])})
    if variant in {"prox", "all_three"}:
        params.update({"use_prox": True, "mu": random_choice(rng, fednova_mu_choices)})
    if variant in {"server_momentum", "both_momentums", "all_three"}:
        params.update({"use_server_momentum": True, "gamma": random_choice(rng, [0.5, 0.9])})
    return candidate("fednova", "FedNova", "FedNova", f"{variant}_random_{index:02d}", "random", params)


def random_fedopt_candidate(rng: np.random.Generator, index: int) -> Candidate:
    algorithm_name = random_choice(rng, ["FedAdam", "FedYogi", "FedAdagrad"])
    params: dict[str, Any] = {
        "step_size": log_uniform(rng, 1e-4, 5e-2),
        "num_local_epochs": random_choice(rng, local_epoch_choices),
        "server_step_size": log_uniform(rng, 1e-4, 5e-2),
        "beta_1": random_choice(rng, [0.0, 0.5, 0.9]),
        "tau": random_choice(rng, [1e-6, 1e-4, 1e-3]),
    }
    if algorithm_name in {"FedAdam", "FedYogi"}:
        params["beta_2"] = random_choice(rng, [0.9, 0.99, 0.999])
    return candidate("fedopt", "FedOpt", algorithm_name, f"random_{index:02d}", "random", params)


def random_fedlt_candidate(rng: np.random.Generator, index: int) -> Candidate:
    solver = random_choice(rng, ["gd", "nesterov", "adam"])
    params: dict[str, Any] = {
        "step_size": log_uniform(rng, 1e-4, 1e-2),
        "num_local_epochs": random_choice(rng, local_epoch_choices),
        "rho": log_uniform(rng, 1e-2, 1e1),
        "local_solver": solver,
    }
    if solver == "adam":
        params["solver_args"] = {"beta1": random_choice(rng, [0.5, 0.9]), "beta2": 0.999, "epsilon": 1e-8}
    if solver == "nesterov":
        params["solver_args"] = {"momentum": random_choice(rng, [0.5, 0.9])}
    return candidate("fedlt", "FedLT", "FedLT", f"{solver}_random_{index:02d}", "random", params)


def random_feddyn_candidate(rng: np.random.Generator, index: int) -> Candidate:
    return candidate(
        "feddyn",
        "FedDyn",
        "FedDyn",
        f"random_{index:02d}",
        "random",
        {
            "step_size": log_uniform(rng, 1e-4, 5e-2),
            "num_local_epochs": random_choice(rng, local_epoch_choices),
            "alpha": log_uniform(rng, 1e-2, 1e0),
        },
    )


def random_fedpd_candidate(rng: np.random.Generator, index: int) -> Candidate:
    return candidate(
        "fedpd",
        "FedPD",
        "FedPD",
        f"random_{index:02d}",
        "random",
        {
            "step_size": log_uniform(rng, 1e-4, 5e-2),
            "num_local_epochs": random_choice(rng, local_epoch_choices),
            "eta": log_uniform(rng, 1e-2, 1e0),
            "skip_probability": random_choice(rng, [0.0, 0.1, 0.2]),
        },
    )


def grid_candidates_from(best: Candidate) -> list[Candidate]:
    params = best.hyperparameters
    common_grid = list(
        product(
            nearby_log_values(float(params["step_size"]), lower=1e-4, upper=5e-2),
            nearby_epoch_values(int(params["num_local_epochs"])),
        )
    )
    if best.algorithm_key == "fedavg":
        return [
            candidate("fedavg", "FedAvg", "FedAvg", f"grid_lr_{format_value(step)}_e{epochs}", "grid", {"step_size": step, "num_local_epochs": epochs})
            for step, epochs in common_grid
        ]
    if best.algorithm_key == "fedprox":
        return [
            candidate("fedprox", "FedProx", "FedProx", f"grid_lr_{format_value(step)}_e{epochs}_mu_{format_value(mu)}", "grid", {"step_size": step, "num_local_epochs": epochs, "mu": mu})
            for step, epochs in common_grid
            for mu in nearby_log_values(float(params["mu"]), lower=1e-4, upper=1e-1)
        ]
    if best.algorithm_key == "scaffold":
        return [
            candidate("scaffold", "SCAFFOLD", "Scaffold", f"grid_lr_{format_value(step)}_e{epochs}_server_{format_value(server_step)}", "grid", {"step_size": step, "num_local_epochs": epochs, "server_step_size": server_step})
            for step, epochs in common_grid
            for server_step in nearby_linear_values(float(params["server_step_size"]), 0.5, 1.0)
        ]
    if best.algorithm_key == "fedopt":
        grid = []
        for step, epochs in common_grid:
            for server_step in nearby_log_values(float(params["server_step_size"]), lower=1e-4, upper=5e-2):
                candidate_params = {
                    "step_size": step,
                    "num_local_epochs": epochs,
                    "server_step_size": server_step,
                    "beta_1": params["beta_1"],
                    "tau": params["tau"],
                }
                if best.algorithm_name in {"FedAdam", "FedYogi"}:
                    candidate_params["beta_2"] = params["beta_2"]
                grid.append(candidate("fedopt", "FedOpt", best.algorithm_name, f"grid_lr_{format_value(step)}_e{epochs}", "grid", candidate_params))
        return grid
    base = {key: value for key, value in params.items() if key not in {"step_size", "num_local_epochs"}}
    return [
        candidate(best.algorithm_key, best.group, best.algorithm_name, f"grid_lr_{format_value(step)}_e{epochs}", "grid", {**base, "step_size": step, "num_local_epochs": epochs})
        for step, epochs in common_grid
    ]


def deduplicate_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    seen = set()
    unique = []
    for tuning_candidate in candidates:
        key = (tuning_candidate.algorithm_name, json.dumps(tuning_candidate.hyperparameters, sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(tuning_candidate)
    return unique


def limit_grid_candidates(candidates: Sequence[Candidate], *, max_candidates: int, rng: np.random.Generator) -> list[Candidate]:
    unique = deduplicate_candidates(candidates)
    if len(unique) <= max_candidates:
        return unique
    selected_indices = sorted(rng.choice(len(unique), size=max_candidates, replace=False).tolist())
    return [unique[index] for index in selected_indices]


def split_train_validation(train_partitions: Sequence[Sequence[Datapoint]], *, seed: int) -> tuple[list[IndexedSubset], ConcatDataset]:
    iop.set_seed(seed)
    tuning_partitions: list[IndexedSubset] = []
    validation_subsets: list[IndexedSubset] = []
    for partition in train_partitions:
        indices = iop.rng_numpy().permutation(len(partition)).tolist()
        n_validation = max(1, round(len(partition) * validation_fraction))
        validation_indices = sorted(indices[:n_validation])
        train_indices = sorted(indices[n_validation:])
        labels = getattr(partition, "labels", None)
        train_labels = [labels[index] for index in train_indices] if labels is not None else None
        validation_labels = [labels[index] for index in validation_indices] if labels is not None else None
        tuning_partitions.append(IndexedSubset(partition, train_indices, labels=train_labels))
        validation_subsets.append(IndexedSubset(partition, validation_indices, labels=validation_labels))
    return tuning_partitions, ConcatDataset(validation_subsets)


def build_problem(
    train_partitions: Sequence[Sequence[Datapoint]],
    validation_data: Sequence[Datapoint],
    center_ids: Sequence[int],
    *,
    config: RuntimeConfig,
    state_snapshot_period: int,
) -> tuple[benchmark.BenchmarkProblem, Any]:
    iop.set_seed(seed)
    num_classes = infer_num_classes(train_partitions)
    alpha = build_alpha(train_partitions, num_classes, mode=config.class_weight_mode)
    costs = [
        PyTorchCost(
            dataset=partition,
            model=build_model(config.model_name, num_classes=num_classes, pretrained=config.pretrained),
            loss_fn=WeightedFocalLoss(alpha=alpha),
            final_activation=ArgmaxActivation(),
            batch_size=min(config.batch_size, len(partition)),
            max_batch_size=config.batch_size,
            device=config.device,
            load_dataset=config.load_dataset,
        )
        for partition in train_partitions
    ]
    agents = [
        Agent(
            cost,
            activation=AlwaysActive(),
            state_snapshot_period=state_snapshot_period,
            data={"center_id": center_id},
        )
        for center_id, cost in zip(center_ids, costs, strict=True)
    ]
    network = FedNetwork(
        clients=agents,
        message_noise=NoNoise(),
        message_compression=NoCompression(),
        message_drop=NoDrops(),
    )
    problem = benchmark.BenchmarkProblem(network=network, test_data=validation_data)
    x0 = pytorch_initialization(network, all_same=True)
    return problem, x0


def run_candidate(
    tuning_candidate: Candidate,
    train_partitions: Sequence[Sequence[Datapoint]],
    validation_data: Sequence[Datapoint],
    center_ids: Sequence[int],
    *,
    config: RuntimeConfig,
    iterations: int,
    state_snapshot_period: int,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    problem, x0 = build_problem(
        train_partitions,
        validation_data,
        center_ids,
        config=config,
        state_snapshot_period=state_snapshot_period,
    )
    algorithm = build_algorithm(tuning_candidate, x0, iterations)
    result = benchmark.benchmark(
        algorithms=[algorithm],
        benchmark_problem=problem,
        n_trials=config.n_trials,
        max_processes=1,
        progress_step=max(1, iterations // 10),
        show_speed=True,
        show_trial=True,
        checkpoint_manager=None,
        log_level=logging.INFO,
    )
    metric_result = benchmark.compute_metrics(
        benchmark_result=result,
        table_metrics=[
            ml.ServerBalancedAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.BalancedAccuracy([np.average], fmt=".2%", x_log=False, y_log=False),
            ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
            ml.Loss([np.average]),
        ],
        plot_metrics=[],
        log_level=logging.INFO,
    )
    table_frame, _ = metric_result.to_dataframe()
    if table_frame is None:
        raise RuntimeError("No table metrics were computed.")

    server_bal_acc, server_bal_acc_ci = metric_value(table_frame, "server balanced accuracy")
    server_acc, _ = metric_value(table_frame, "server accuracy")
    validation_loss, validation_loss_ci = metric_value(table_frame, "loss", statistic="avg")
    return {
        "status": "ok",
        "search_stage": tuning_candidate.search_stage,
        "candidate_id": tuning_candidate.candidate_id,
        "algorithm_key": tuning_candidate.algorithm_key,
        "algorithm_group": tuning_candidate.group,
        "algorithm_name": tuning_candidate.algorithm_name,
        "variant": tuning_candidate.variant,
        "server_balanced_accuracy_mean": server_bal_acc,
        "server_balanced_accuracy_margin_of_error": server_bal_acc_ci,
        "server_accuracy_mean": server_acc,
        "validation_loss_mean": validation_loss,
        "validation_loss_margin_of_error": validation_loss_ci,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        **flatten_hyperparameters(tuning_candidate.hyperparameters),
    }


def run_final_curve(
    best_candidate: Candidate,
    train_partitions: Sequence[Sequence[Datapoint]],
    validation_data: Sequence[Datapoint],
    center_ids: Sequence[int],
    *,
    config: RuntimeConfig,
) -> None:
    final_path = config.run_path / "final_best_candidate_curve"
    final_path.mkdir(parents=True, exist_ok=True)
    state_snapshot_period = max(1, config.final_iterations // 10)
    problem, x0 = build_problem(
        train_partitions,
        validation_data,
        center_ids,
        config=config,
        state_snapshot_period=state_snapshot_period,
    )
    algorithm = build_algorithm(best_candidate, x0, config.final_iterations)
    result = benchmark.benchmark(
        algorithms=[algorithm],
        benchmark_problem=problem,
        n_trials=config.n_trials,
        max_processes=1,
        progress_step=max(1, config.final_iterations // 10),
        show_speed=True,
        show_trial=True,
        checkpoint_manager=None,
        log_level=logging.INFO,
    )
    metric_result = benchmark.compute_metrics(
        benchmark_result=result,
        table_metrics=[
            ml.ServerBalancedAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.BalancedAccuracy([np.average], fmt=".2%", x_log=False, y_log=False),
            ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.Loss([np.average]),
        ],
        plot_metrics=[
            ml.ServerBalancedAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.Loss([np.average]),
        ],
        log_level=logging.INFO,
    )
    benchmark.display_metrics(metrics_result=metric_result, save_path=final_path / "results", show_plots=False)
    metric_result.agent_metrics = None
    save_pickle_zst(metric_result, final_path / metric_result_filename)
    (final_path / "metadata.json").write_text(
        json.dumps(
            {
                "final_iterations": config.final_iterations,
                "state_snapshot_period": state_snapshot_period,
                "n_trials": config.n_trials,
                "best_candidate": best_payload(best_candidate, row=None),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_data(config: RuntimeConfig) -> tuple[list[IndexedSubset], ConcatDataset, list[int], dict[str, Any]]:
    iop.set_seed(seed)
    train_dataset = FedISICDatasetHandler(
        split="train",
        max_samples_per_client=config.max_samples_per_client,
        local_files_only=config.local_files_only,
    )
    train_partitions, validation_data = split_train_validation(train_dataset.get_partitions(), seed=seed)
    metadata = {
        "center_ids": train_dataset.center_ids,
        "center_names": train_dataset.center_names,
        "class_names": train_dataset.class_names,
        "n_classes": train_dataset.n_targets,
        "n_train_samples_after_validation_split": sum(len(partition) for partition in train_partitions),
        "n_validation_samples": len(validation_data),
        "max_samples_per_client": config.max_samples_per_client,
    }
    return train_partitions, validation_data, train_dataset.center_ids, metadata


def run_candidate_list(
    candidates: Sequence[Candidate],
    train_partitions: Sequence[Sequence[Datapoint]],
    validation_data: Sequence[Datapoint],
    center_ids: Sequence[int],
    *,
    config: RuntimeConfig,
    candidate_results_path: Path,
    starting_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, tuning_candidate in enumerate(candidates, start=starting_index):
        print(f"[{offset}] {tuning_candidate.candidate_id}")
        try:
            row = run_candidate(
                tuning_candidate,
                train_partitions,
                validation_data,
                center_ids,
                config=config,
                iterations=config.iterations,
                state_snapshot_period=config.iterations,
            )
        except Exception as exc:
            logging.exception("Candidate failed: %s", tuning_candidate.candidate_id)
            row = {
                "status": "failed",
                "search_stage": tuning_candidate.search_stage,
                "candidate_id": tuning_candidate.candidate_id,
                "algorithm_key": tuning_candidate.algorithm_key,
                "algorithm_group": tuning_candidate.group,
                "algorithm_name": tuning_candidate.algorithm_name,
                "variant": tuning_candidate.variant,
                "error": repr(exc),
                **flatten_hyperparameters(tuning_candidate.hyperparameters),
            }
        rows.append(row)
        append_candidate_result(row, candidate_results_path)
        cleanup_cuda()
    return rows


def best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["status"] == "ok"]
    if not valid_rows:
        raise RuntimeError("No successful candidates were available to select from.")
    return max(valid_rows, key=lambda row: (row["server_balanced_accuracy_mean"], -row["validation_loss_mean"]))


def is_better(candidate_row: dict[str, Any], current_row: dict[str, Any]) -> bool:
    if candidate_row["server_balanced_accuracy_mean"] != current_row["server_balanced_accuracy_mean"]:
        return candidate_row["server_balanced_accuracy_mean"] > current_row["server_balanced_accuracy_mean"]
    return candidate_row["validation_loss_mean"] < current_row["validation_loss_mean"]


def best_rows_by_algorithm_name(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_name: dict[str, dict[str, Any]] = {}
    for row in [candidate_row for candidate_row in rows if candidate_row["status"] == "ok"]:
        current = best_by_name.get(row["algorithm_name"])
        if current is None or is_better(row, current):
            best_by_name[row["algorithm_name"]] = row
    return list(best_by_name.values())


def row_to_candidate(row: dict[str, Any], candidates: Sequence[Candidate]) -> Candidate:
    for tuning_candidate in candidates:
        if tuning_candidate.candidate_id == row["candidate_id"]:
            return tuning_candidate
    raise RuntimeError(f"Could not find candidate for row {row['candidate_id']!r}.")


def save_best_hyperparameters(best_candidate: Candidate, best_result_row: dict[str, Any], path: Path, *, config: RuntimeConfig) -> None:
    payload = {
        "metadata": {
            "experiment": "experiment0",
            "dataset": "Fed-ISIC2019",
            "dataset_source": "flwrlabs/fed-isic2019",
            "partition": "natural FLamby/Flower center split",
            "n_clients": 6,
            "validation_fraction_from_train": validation_fraction,
            "n_trials": config.n_trials,
            "iterations": config.iterations,
            "batch_size": config.batch_size,
            "seed": seed,
            "selection_metric": selection_metric,
            "tie_break_metric": tie_break_metric,
            "client_participation": "full",
            "communication": "no drops, no noise, no compression",
            "model": config.model_name,
            "pretrained": config.pretrained if config.model_name == "efficientnet_b0" else False,
            "class_weight_mode": config.class_weight_mode,
            "search_strategy": "reference candidates plus random coarse search followed by focused grid search",
        },
        "best_hyperparameters": best_payload(best_candidate, best_result_row),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def best_payload(best_candidate: Candidate, row: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "algorithm_name": best_candidate.algorithm_name,
        "variant": best_candidate.variant,
        "search_stage": best_candidate.search_stage,
        "hyperparameters": best_candidate.hyperparameters,
    }
    if row is not None:
        payload.update(
            {
                "server_balanced_accuracy_mean": row["server_balanced_accuracy_mean"],
                "server_balanced_accuracy_margin_of_error": row["server_balanced_accuracy_margin_of_error"],
                "server_accuracy_mean": row["server_accuracy_mean"],
                "validation_loss_mean": row["validation_loss_mean"],
                "validation_loss_margin_of_error": row["validation_loss_margin_of_error"],
            }
        )
    if best_candidate.group == "FedOpt":
        payload["selected_algorithm"] = best_candidate.algorithm_name
    if best_candidate.group == "FedLT":
        payload["selected_solver"] = best_candidate.hyperparameters["local_solver"]
    return payload


def metric_value(table_frame: Any, metric_name: str, *, statistic: str | None = None) -> tuple[float, float]:
    rows = table_frame[table_frame["metric"] == metric_name]
    if statistic is not None:
        statistic_rows = rows[rows["statistic"] == statistic]
        if not statistic_rows.empty:
            rows = statistic_rows
    if rows.empty:
        raise RuntimeError(f"Metric {metric_name!r} with statistic {statistic!r} was not computed.")
    row = rows.iloc[0]
    return float(row["mean"]), float(row["margin_of_error"])


def append_candidate_result(row: dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_result_fields, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def flatten_hyperparameters(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in hyperparameters.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat[f"{key}.{nested_key}"] = nested_value
        else:
            flat[key] = value
    return flat


def infer_num_classes(partitions: Sequence[Sequence[Datapoint]]) -> int:
    labels = []
    for partition in partitions:
        labels.extend(labels_from_dataset(partition))
    return max(labels) + 1


def labels_from_dataset(dataset: Sequence[Datapoint]) -> list[int]:
    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return [int(label) for label in labels]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def build_alpha(partitions: Sequence[Sequence[Datapoint]], num_classes: int, *, mode: str) -> torch.Tensor | None:
    if mode == "flamby":
        return None
    labels = []
    for partition in partitions:
        labels.extend(labels_from_dataset(partition))
    return class_weights_from_labels(labels, num_classes)


def log_uniform(rng: np.random.Generator, lower: float, upper: float) -> float:
    return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))


def random_choice(rng: np.random.Generator, choices: Sequence[Any]) -> Any:
    return choices[int(rng.integers(0, len(choices)))]


def nearby_log_values(value: float, *, lower: float, upper: float) -> list[float]:
    values = [value / 2.0, value, value * 2.0]
    return sorted({float(min(max(candidate_value, lower), upper)) for candidate_value in values})


def nearby_linear_values(value: float, lower: float, upper: float) -> list[float]:
    delta = max((upper - lower) / 4.0, 1e-12)
    values = [value - delta, value, value + delta]
    return sorted({float(min(max(candidate_value, lower), upper)) for candidate_value in values})


def nearby_epoch_values(value: int) -> list[int]:
    return sorted({epoch for epoch in (value - 1, value, value + 1) if epoch > 0 and epoch <= max(local_epoch_choices)})


def format_value(value: float) -> str:
    return f"{value:.3g}".replace(".", "p").replace("-", "m")


def _group_name(algorithm_key: str) -> str:
    return {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "scaffold": "SCAFFOLD",
        "fednova": "FedNova",
        "fedopt": "FedOpt",
        "fedlt": "FedLT",
        "feddyn": "FedDyn",
        "fedpd": "FedPD",
    }[algorithm_key]


def _default_algorithm_name(algorithm_key: str) -> str:
    return {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "scaffold": "Scaffold",
        "fednova": "FedNova",
        "fedopt": "FedAdam",
        "fedlt": "FedLT",
        "feddyn": "FedDyn",
        "fedpd": "FedPD",
    }[algorithm_key]


def save_pickle_zst(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor(level=1)
    with path.open("wb") as file_obj, compressor.stream_writer(file_obj) as compressed_writer:
        pickle.dump(data, compressed_writer, protocol=pickle.HIGHEST_PROTOCOL)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    config = parse_args()
    config.run_path.mkdir(parents=True, exist_ok=True)
    candidate_results_path = config.run_path / "exp0_candidate_results.csv"
    best_path = config.run_path / "exp0_best_hyperparameters.json"

    train_partitions, validation_data, center_ids, data_metadata = load_data(config)
    run_metadata = {
        **data_metadata,
        "run_config": {
            "algorithm": config.algorithm,
            "iterations": config.iterations,
            "final_iterations": config.final_iterations,
            "n_trials": config.n_trials,
            "n_random_candidates": config.n_random_candidates,
            "max_grid_candidates": config.max_grid_candidates,
            "run_final": config.run_final,
            "batch_size": config.batch_size,
            "device": config.device.value,
            "model": config.model_name,
            "pretrained": config.pretrained,
            "class_weight_mode": config.class_weight_mode,
            "client_participation": "full",
        },
    }
    (config.run_path / "exp0_dataset_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print(f"Experiment 0 run path: {config.run_path}")
    print(f"Algorithm: {config.algorithm}")
    print("Candidate tuning uses full client participation and clean communication.")

    rng = np.random.default_rng(seed)
    coarse_candidates = deduplicate_candidates(
        [*reference_candidates(config.algorithm), *random_candidates(config.algorithm, rng, config.n_random_candidates)]
    )
    print(f"Running reference/random coarse search with {len(coarse_candidates)} candidates.")
    all_candidates = list(coarse_candidates)
    all_rows = run_candidate_list(
        coarse_candidates,
        train_partitions,
        validation_data,
        center_ids,
        config=config,
        candidate_results_path=candidate_results_path,
        starting_index=1,
    )

    if config.algorithm == "fedopt":
        grid_candidates = []
        for fedopt_best_row in best_rows_by_algorithm_name(all_rows):
            grid_candidates.extend(grid_candidates_from(row_to_candidate(fedopt_best_row, coarse_candidates)))
        grid_candidates = limit_grid_candidates(grid_candidates, max_candidates=config.max_grid_candidates, rng=rng)
    else:
        grid_candidates = limit_grid_candidates(
            grid_candidates_from(row_to_candidate(best_row(all_rows), coarse_candidates)),
            max_candidates=config.max_grid_candidates,
            rng=rng,
        )

    print(f"Running focused grid search with {len(grid_candidates)} candidates.")
    all_candidates.extend(grid_candidates)
    grid_rows = run_candidate_list(
        grid_candidates,
        train_partitions,
        validation_data,
        center_ids,
        config=config,
        candidate_results_path=candidate_results_path,
        starting_index=len(all_rows) + 1,
    )
    all_rows.extend(grid_rows)

    final_best_row = best_row(all_rows)
    final_best_candidate = row_to_candidate(final_best_row, all_candidates)
    save_best_hyperparameters(final_best_candidate, final_best_row, best_path, config=config)

    if config.run_final:
        print(f"Running final best-candidate curve for {config.final_iterations} iterations.")
        run_final_curve(final_best_candidate, train_partitions, validation_data, center_ids, config=config)

    print(f"Candidate results saved to: {candidate_results_path}")
    print(f"Best hyperparameters saved to: {best_path}")


if __name__ == "__main__":
    main()
