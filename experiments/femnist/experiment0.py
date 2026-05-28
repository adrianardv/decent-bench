from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import decent_bench.utils.interoperability as iop
import numpy as np
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
from decent_bench.metrics import metric_library as ml
from decent_bench.networks import FedNetwork
from decent_bench.schemes import AlwaysActive, NoCompression, NoDrops, NoNoise, UniformSelection
from decent_bench.utils.pytorch_utils import ArgmaxActivation
from decent_bench.utils.types import Dataset, SupportedDevices

from experiments.femnist.src import FEMNISTCNN, FEMNISTDatasetHandler


# -----------------------------------------------------------------------------
# Fixed FEMNIST experiment setup
# -----------------------------------------------------------------------------

seed = 20260524
n_clients = 100
min_train_samples = 100
min_test_samples = 20
train_fraction = 0.8
validation_fraction = 0.2
batch_size = 32
device = SupportedDevices.GPU
local_files_only = False
load_dataset = True
selection_fraction = 0.2
selection_metric = "server accuracy"
tie_break_metric = "loss"
local_epoch_choices = [1, 2, 3, 5, 8, 10]

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

candidate_result_fields = [
    "status",
    "search_stage",
    "candidate_id",
    "algorithm_key",
    "algorithm_group",
    "algorithm_name",
    "variant",
    "server_accuracy_mean",
    "server_accuracy_margin_of_error",
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


# -----------------------------------------------------------------------------
# Runtime and candidate data structures
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Tune FEMNIST hyperparameters for one federated algorithm.")
    parser.add_argument("--algorithm", choices=algorithm_choices, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--final-iterations", type=int, default=2000)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--n-random-candidates", type=int, default=12)
    parser.add_argument("--max-grid-candidates", type=int, default=27)
    parser.add_argument("--skip-final-run", action="store_true")
    parser.add_argument("--run-name", type=str)
    args = parser.parse_args()

    run_name = args.run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_path = Path("experiments/femnist/checkpoints/experiment0") / args.algorithm / run_name
    return RuntimeConfig(
        algorithm=args.algorithm,
        iterations=args.iterations,
        final_iterations=args.final_iterations,
        n_trials=args.n_trials,
        n_random_candidates=args.n_random_candidates,
        max_grid_candidates=args.max_grid_candidates,
        run_final=not args.skip_final_run,
        run_path=run_path,
    )


# -----------------------------------------------------------------------------
# Algorithm construction
# -----------------------------------------------------------------------------

def selection_scheme_for(algorithm_key: str) -> UniformSelection | None:
    if algorithm_key == "fedpd":
        return None
    return UniformSelection(fraction_selected_clients=selection_fraction)


def build_algorithm(tuning_candidate: Candidate, x0: Any, iterations: int) -> Any:
    params = dict(tuning_candidate.hyperparameters)
    algorithm_key = tuning_candidate.algorithm_key
    algorithm_name = tuning_candidate.algorithm_name

    if algorithm_key == "fedavg":
        return FedAvg(iterations=iterations, selection_scheme=selection_scheme_for(algorithm_key), x0=x0, **params)
    if algorithm_key == "fedprox":
        return FedProx(iterations=iterations, selection_scheme=selection_scheme_for(algorithm_key), x0=x0, **params)
    if algorithm_key == "scaffold":
        return Scaffold(iterations=iterations, selection_scheme=selection_scheme_for(algorithm_key), x0=x0, **params)
    if algorithm_key == "fednova":
        params["num_local_steps"] = params.pop("num_local_epochs")
        return FedNova(iterations=iterations, selection_scheme=selection_scheme_for(algorithm_key), x0=x0, **params)
    if algorithm_key == "fedopt":
        fedopt_class = {"FedAdam": FedAdam, "FedYogi": FedYogi, "FedAdagrad": FedAdagrad}[algorithm_name]
        return fedopt_class(iterations=iterations, selection_scheme=selection_scheme_for(algorithm_key), x0=x0, **params)
    if algorithm_key == "fedlt":
        return FedLT(iterations=iterations, selection_scheme=selection_scheme_for(algorithm_key), x0=x0, **params)
    if algorithm_key == "feddyn":
        return FedDyn(iterations=iterations, selection_scheme=selection_scheme_for(algorithm_key), x0=x0, **params)
    if algorithm_key == "fedpd":
        params["num_local_steps"] = params.pop("num_local_epochs")
        return FedPD(iterations=iterations, x0=x0, **params)
    raise ValueError(f"Unsupported algorithm key: {algorithm_key}")


# -----------------------------------------------------------------------------
# Candidate generation
# -----------------------------------------------------------------------------


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
    params = {
        "step_size": log_uniform(rng, 1e-3, 1e-1),
        "num_local_epochs": random_choice(rng, local_epoch_choices),
    }
    return candidate("fedavg", "FedAvg", "FedAvg", f"random_{index:02d}", "random", params)


def random_fedprox_candidate(rng: np.random.Generator, index: int) -> Candidate:
    params = {
        "step_size": log_uniform(rng, 1e-3, 1e-1),
        "num_local_epochs": random_choice(rng, local_epoch_choices),
        "mu": log_uniform(rng, 1e-4, 1e0),
    }
    return candidate("fedprox", "FedProx", "FedProx", f"random_{index:02d}", "random", params)


def random_scaffold_candidate(rng: np.random.Generator, index: int) -> Candidate:
    params = {
        "step_size": log_uniform(rng, 1e-3, 5e-2),
        "num_local_epochs": random_choice(rng, local_epoch_choices),
        "server_step_size": random_choice(rng, [0.1, 0.5, 1.0]),
    }
    return candidate("scaffold", "SCAFFOLD", "Scaffold", f"random_{index:02d}", "random", params)


def random_fednova_candidate(rng: np.random.Generator, index: int) -> Candidate:
    variant = random_choice(rng, ["plain", "momentum", "prox", "server_momentum"])
    params: dict[str, Any] = {
        "step_size": log_uniform(rng, 1e-3, 5e-2),
        "num_local_epochs": random_choice(rng, local_epoch_choices),
    }
    if variant == "momentum":
        params.update({"use_momentum": True, "beta": random_choice(rng, [0.5, 0.9])})
    elif variant == "prox":
        params.update({"use_prox": True, "mu": log_uniform(rng, 1e-4, 1e-1)})
    elif variant == "server_momentum":
        params.update({"use_server_momentum": True, "gamma": random_choice(rng, [0.5, 0.9])})
    return candidate("fednova", "FedNova", "FedNova", f"{variant}_random_{index:02d}", "random", params)


def random_fedopt_candidate(rng: np.random.Generator, index: int) -> Candidate:
    algorithm_name = random_choice(rng, ["FedAdam", "FedYogi", "FedAdagrad"])
    params: dict[str, Any] = {
        "step_size": log_uniform(rng, 1e-3, 5e-2),
        "num_local_epochs": random_choice(rng, local_epoch_choices),
        "server_step_size": log_uniform(rng, 1e-4, 1e-1),
        "beta_1": random_choice(rng, [0.0, 0.5, 0.9]),
        "tau": random_choice(rng, [1e-6, 1e-4, 1e-3]),
    }
    if algorithm_name in {"FedAdam", "FedYogi"}:
        params["beta_2"] = random_choice(rng, [0.9, 0.99, 0.999])
    return candidate("fedopt", "FedOpt", algorithm_name, f"random_{index:02d}", "random", params)


def random_fedlt_candidate(rng: np.random.Generator, index: int) -> Candidate:
    params = {
        "step_size": log_uniform(rng, 1e-3, 5e-2),
        "num_local_epochs": random_choice(rng, local_epoch_choices),
        "rho": log_uniform(rng, 1e-2, 1e1),
        "local_solver": "gd",
    }
    return candidate("fedlt", "FedLT", "FedLT", f"gd_random_{index:02d}", "random", params)


def random_feddyn_candidate(rng: np.random.Generator, index: int) -> Candidate:
    params = {
        "step_size": log_uniform(rng, 1e-3, 5e-2),
        "num_local_epochs": random_choice(rng, local_epoch_choices),
        "alpha": log_uniform(rng, 1e-4, 1e0),
    }
    return candidate("feddyn", "FedDyn", "FedDyn", f"random_{index:02d}", "random", params)


def random_fedpd_candidate(rng: np.random.Generator, index: int) -> Candidate:
    params = {
        "step_size": log_uniform(rng, 1e-3, 5e-2),
        "num_local_epochs": random_choice(rng, local_epoch_choices),
        "eta": log_uniform(rng, 1e-2, 1e1),
        "skip_probability": random_choice(rng, [0.0, 0.1, 0.2]),
    }
    return candidate("fedpd", "FedPD", "FedPD", f"random_{index:02d}", "random", params)


def grid_candidates_from(best: Candidate) -> list[Candidate]:
    params = best.hyperparameters
    common_grid = list(
        product(
            nearby_log_values(float(params["step_size"]), lower=1e-4, upper=2e-1),
            nearby_epoch_values(int(params["num_local_epochs"])),
        )
    )
    builders = {
        "fedavg": grid_fedavg_candidates,
        "fedprox": grid_fedprox_candidates,
        "scaffold": grid_scaffold_candidates,
        "fednova": grid_fednova_candidates,
        "fedopt": grid_fedopt_candidates,
        "fedlt": grid_fedlt_candidates,
        "feddyn": grid_feddyn_candidates,
        "fedpd": grid_fedpd_candidates,
    }
    return deduplicate_candidates(builders[best.algorithm_key](best, common_grid))


def grid_fedavg_candidates(best: Candidate, common_grid: list[tuple[float, int]]) -> list[Candidate]:
    return [
        candidate(
            "fedavg",
            "FedAvg",
            "FedAvg",
            f"grid_lr_{format_value(step)}_e{epochs}",
            "grid",
            {"step_size": step, "num_local_epochs": epochs},
        )
        for step, epochs in common_grid
    ]


def grid_fedprox_candidates(best: Candidate, common_grid: list[tuple[float, int]]) -> list[Candidate]:
    mu_values = nearby_log_values(float(best.hyperparameters["mu"]), lower=1e-5, upper=1e1)
    return [
        candidate(
            "fedprox",
            "FedProx",
            "FedProx",
            f"grid_lr_{format_value(step)}_e{epochs}_mu_{format_value(mu)}",
            "grid",
            {"step_size": step, "num_local_epochs": epochs, "mu": mu},
        )
        for step, epochs in common_grid
        for mu in mu_values
    ]


def grid_scaffold_candidates(best: Candidate, common_grid: list[tuple[float, int]]) -> list[Candidate]:
    server_values = nearby_linear_values(float(best.hyperparameters["server_step_size"]), lower=0.05, upper=1.5)
    return [
        candidate(
            "scaffold",
            "SCAFFOLD",
            "Scaffold",
            f"grid_lr_{format_value(step)}_e{epochs}_server_{format_value(server_step)}",
            "grid",
            {"step_size": step, "num_local_epochs": epochs, "server_step_size": server_step},
        )
        for step, epochs in common_grid
        for server_step in server_values
    ]


def grid_fednova_candidates(best: Candidate, common_grid: list[tuple[float, int]]) -> list[Candidate]:
    base = {
        key: value
        for key, value in best.hyperparameters.items()
        if key not in {"step_size", "num_local_epochs", "beta", "mu", "gamma"}
    }
    extra_grids: list[dict[str, Any]] = [{}]
    if best.hyperparameters.get("use_momentum"):
        extra_grids = [{"beta": value} for value in nearby_linear_values(float(best.hyperparameters["beta"]), 0.0, 0.99)]
    elif best.hyperparameters.get("use_prox"):
        extra_grids = [{"mu": value} for value in nearby_log_values(float(best.hyperparameters["mu"]), 1e-5, 1e0)]
    elif best.hyperparameters.get("use_server_momentum"):
        extra_grids = [
            {"gamma": value} for value in nearby_linear_values(float(best.hyperparameters["gamma"]), 0.0, 0.99)
        ]

    return [
        candidate(
            "fednova",
            "FedNova",
            "FedNova",
            f"grid_{best.variant}_lr_{format_value(step)}_e{epochs}_{grid_index}",
            "grid",
            {**base, **extra, "step_size": step, "num_local_epochs": epochs},
        )
        for step, epochs in common_grid
        for grid_index, extra in enumerate(extra_grids)
    ]


def grid_fedopt_candidates(best: Candidate, common_grid: list[tuple[float, int]]) -> list[Candidate]:
    server_values = nearby_log_values(float(best.hyperparameters["server_step_size"]), lower=1e-5, upper=1e0)
    candidates: list[Candidate] = []
    for step, epochs in common_grid:
        for server_step in server_values:
            params: dict[str, Any] = {
                "step_size": step,
                "num_local_epochs": epochs,
                "server_step_size": server_step,
                "beta_1": best.hyperparameters["beta_1"],
                "tau": best.hyperparameters["tau"],
            }
            if best.algorithm_name in {"FedAdam", "FedYogi"}:
                params["beta_2"] = best.hyperparameters["beta_2"]
            candidates.append(
                candidate(
                    "fedopt",
                    "FedOpt",
                    best.algorithm_name,
                    f"grid_{best.algorithm_name}_lr_{format_value(step)}_e{epochs}",
                    "grid",
                    params,
                )
            )
    return candidates


def grid_fedlt_candidates(best: Candidate, common_grid: list[tuple[float, int]]) -> list[Candidate]:
    rho_values = nearby_log_values(float(best.hyperparameters["rho"]), lower=1e-3, upper=1e2)
    return [
        candidate(
            "fedlt",
            "FedLT",
            "FedLT",
            f"gd_grid_lr_{format_value(step)}_e{epochs}_rho_{format_value(rho)}",
            "grid",
            {"step_size": step, "num_local_epochs": epochs, "rho": rho, "local_solver": "gd"},
        )
        for step, epochs in common_grid
        for rho in rho_values
    ]


def grid_feddyn_candidates(best: Candidate, common_grid: list[tuple[float, int]]) -> list[Candidate]:
    alpha_values = nearby_log_values(float(best.hyperparameters["alpha"]), lower=1e-5, upper=1e1)
    return [
        candidate(
            "feddyn",
            "FedDyn",
            "FedDyn",
            f"grid_lr_{format_value(step)}_e{epochs}_alpha_{format_value(alpha)}",
            "grid",
            {"step_size": step, "num_local_epochs": epochs, "alpha": alpha},
        )
        for step, epochs in common_grid
        for alpha in alpha_values
    ]


def grid_fedpd_candidates(best: Candidate, common_grid: list[tuple[float, int]]) -> list[Candidate]:
    eta_values = nearby_log_values(float(best.hyperparameters["eta"]), lower=1e-3, upper=1e2)
    skip_values = sorted({0.0, 0.1, float(best.hyperparameters["skip_probability"])})
    return [
        candidate(
            "fedpd",
            "FedPD",
            "FedPD",
            f"grid_lr_{format_value(step)}_e{epochs}_eta_{format_value(eta)}_skip_{format_value(skip)}",
            "grid",
            {"step_size": step, "num_local_epochs": epochs, "eta": eta, "skip_probability": skip},
        )
        for step, epochs in common_grid
        for eta in eta_values
        for skip in skip_values
    ]


def fedlt_solver_candidates(best_gd: Candidate) -> list[Candidate]:
    base = {
        key: value
        for key, value in best_gd.hyperparameters.items()
        if key in {"step_size", "num_local_epochs", "rho"}
    }
    return [
        candidate(
            "fedlt",
            "FedLT",
            "FedLT",
            "solver_gd",
            "solver",
            {**base, "local_solver": "gd"},
        ),
        candidate(
            "fedlt",
            "FedLT",
            "FedLT",
            "solver_adam_default",
            "solver",
            {
                **base,
                "local_solver": "adam",
                "solver_args": {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
            },
        ),
        candidate(
            "fedlt",
            "FedLT",
            "FedLT",
            "solver_nesterov_default",
            "solver",
            {**base, "local_solver": "nesterov", "solver_args": {"momentum": 0.9}},
        ),
    ]


# -----------------------------------------------------------------------------
# Candidate search helpers
# -----------------------------------------------------------------------------

def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10 ** rng.uniform(np.log10(low), np.log10(high)))


def random_choice[T](rng: np.random.Generator, values: Sequence[T]) -> T:
    return values[int(rng.integers(0, len(values)))]


def nearby_log_values(value: float, *, lower: float, upper: float) -> list[float]:
    return sorted({clip_float(value * factor, lower, upper) for factor in (0.5, 1.0, 2.0)})


def nearby_linear_values(value: float, lower: float, upper: float) -> list[float]:
    span = max(abs(value) * 0.5, 0.1)
    return sorted({clip_float(candidate_value, lower, upper) for candidate_value in (value - span, value, value + span)})


def nearby_epoch_values(value: int) -> list[int]:
    return sorted({candidate_value for candidate_value in (value - 1, value, value + 1) if 1 <= candidate_value <= 10})


def clip_float(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def format_value(value: float) -> str:
    return f"{value:.3g}".replace(".", "p").replace("-", "m")


def deduplicate_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen: set[tuple[str, str]] = set()
    unique: list[Candidate] = []
    for tuning_candidate in candidates:
        key = (tuning_candidate.algorithm_name, json.dumps(tuning_candidate.hyperparameters, sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(tuning_candidate)
    return unique


def limit_grid_candidates(
    candidates: Sequence[Candidate],
    *,
    max_candidates: int,
    rng: np.random.Generator,
) -> list[Candidate]:
    if len(candidates) <= max_candidates:
        return list(candidates)
    selected_indices = sorted(rng.choice(len(candidates), size=max_candidates, replace=False).tolist())
    return [candidates[index] for index in selected_indices]


# -----------------------------------------------------------------------------
# Dataset and benchmark problem construction
# -----------------------------------------------------------------------------

def split_train_validation(
    train_partitions: Sequence[Dataset],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[Dataset], Dataset]:
    iop.set_seed(seed)
    tuning_train_partitions: list[Dataset] = []
    validation_data: Dataset = []
    for partition in train_partitions:
        indices = iop.rng_numpy().permutation(len(partition))
        n_validation = max(1, round(len(partition) * validation_fraction))
        validation_indices = set(indices[:n_validation])
        tuning_train_partitions.append(
            [datapoint for index, datapoint in enumerate(partition) if index not in validation_indices]
        )
        validation_data.extend(datapoint for index, datapoint in enumerate(partition) if index in validation_indices)
    return tuning_train_partitions, validation_data


def build_problem(
    train_partitions: Sequence[Dataset],
    validation_data: Dataset,
    selected_writer_ids: Sequence[str],
    *,
    state_snapshot_period: int,
) -> tuple[benchmark.BenchmarkProblem, Any]:
    iop.set_seed(seed)
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
        for writer_id, cost in zip(selected_writer_ids, costs, strict=True)
    ]
    network = FedNetwork(
        clients=agents,
        message_noise=NoNoise(),
        message_compression=NoCompression(),
        message_drop=NoDrops(),
    )
    problem = benchmark.BenchmarkProblem(
        network=network,
        test_data=validation_data,
    )
    x0 = pytorch_initialization(network, all_same=True)
    return problem, x0


# -----------------------------------------------------------------------------
# Benchmark execution and final curve generation
# -----------------------------------------------------------------------------

def run_candidate(
    tuning_candidate: Candidate,
    train_partitions: Sequence[Dataset],
    validation_data: Dataset,
    selected_writer_ids: Sequence[str],
    *,
    iterations: int,
    n_trials: int,
    state_snapshot_period: int,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    problem, x0 = build_problem(
        train_partitions,
        validation_data,
        selected_writer_ids,
        state_snapshot_period=state_snapshot_period,
    )
    algorithm = build_algorithm(tuning_candidate, x0, iterations)
    result = benchmark.benchmark(
        algorithms=[algorithm],
        benchmark_problem=problem,
        n_trials=n_trials,
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

    server_accuracy, server_accuracy_ci = metric_value(table_frame, "server accuracy")
    validation_loss, validation_loss_ci = metric_value(table_frame, "loss", statistic="avg")
    return {
        "status": "ok",
        "search_stage": tuning_candidate.search_stage,
        "candidate_id": tuning_candidate.candidate_id,
        "algorithm_key": tuning_candidate.algorithm_key,
        "algorithm_group": tuning_candidate.group,
        "algorithm_name": tuning_candidate.algorithm_name,
        "variant": tuning_candidate.variant,
        "server_accuracy_mean": server_accuracy,
        "server_accuracy_margin_of_error": server_accuracy_ci,
        "validation_loss_mean": validation_loss,
        "validation_loss_margin_of_error": validation_loss_ci,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        **flatten_hyperparameters(tuning_candidate.hyperparameters),
    }


def run_final_curve(
    best_candidate: Candidate,
    train_partitions: Sequence[Dataset],
    validation_data: Dataset,
    selected_writer_ids: Sequence[str],
    *,
    config: RuntimeConfig,
) -> None:
    final_path = config.run_path / "final_best_candidate_curve"
    final_path.mkdir(parents=True, exist_ok=True)
    state_snapshot_period = max(1, config.final_iterations // 10)
    problem, x0 = build_problem(
        train_partitions,
        validation_data,
        selected_writer_ids,
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
            ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.Accuracy([np.average], fmt=".2%", x_log=False, y_log=False),
            ml.Loss([np.average]),
        ],
        plot_metrics=[
            ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.Loss([np.average]),
        ],
        log_level=logging.INFO,
    )
    benchmark.display_metrics(
        metrics_result=metric_result,
        save_path=final_path / "results",
        show_plots=False,
        log_level=logging.INFO,
    )
    metadata = {
        "final_iterations": config.final_iterations,
        "state_snapshot_period": state_snapshot_period,
        "n_trials": config.n_trials,
        "best_candidate": best_payload(best_candidate, row=None),
    }
    (final_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Result extraction and persistence
# -----------------------------------------------------------------------------

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


def flatten_hyperparameters(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in hyperparameters.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat[f"{key}.{nested_key}"] = nested_value
        else:
            flat[key] = value
    return flat


def append_candidate_result(row: dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_result_fields, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def successful_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["status"] == "ok"]


def best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = successful_rows(rows)
    if not valid_rows:
        raise RuntimeError("No successful candidates were available to select from.")
    return max(valid_rows, key=lambda row: (row["server_accuracy_mean"], -row["validation_loss_mean"]))


def best_rows_by_algorithm_name(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_name: dict[str, dict[str, Any]] = {}
    for row in successful_rows(rows):
        algorithm_name = row["algorithm_name"]
        current = best_by_name.get(algorithm_name)
        if current is None or is_better(row, current):
            best_by_name[algorithm_name] = row
    return list(best_by_name.values())


def is_better(candidate_row: dict[str, Any], current_row: dict[str, Any]) -> bool:
    if candidate_row["server_accuracy_mean"] != current_row["server_accuracy_mean"]:
        return candidate_row["server_accuracy_mean"] > current_row["server_accuracy_mean"]
    return candidate_row["validation_loss_mean"] < current_row["validation_loss_mean"]


def save_best_hyperparameters(
    best_candidate: Candidate,
    best_result_row: dict[str, Any],
    path: Path,
    *,
    config: RuntimeConfig,
) -> None:
    payload = {
        "metadata": {
            "experiment": "experiment0",
            "algorithm": config.algorithm,
            "dataset": "FEMNIST",
            "dataset_source": "flwrlabs/femnist",
            "partition": "natural writer/client split",
            "n_clients": n_clients,
            "min_train_samples": min_train_samples,
            "min_test_samples": min_test_samples,
            "train_fraction": train_fraction,
            "validation_fraction_from_train": validation_fraction,
            "n_trials": config.n_trials,
            "iterations": config.iterations,
            "state_snapshot_period": config.iterations,
            "checkpoint_step": None,
            "batch_size": batch_size,
            "seed": seed,
            "selection_fraction": None if config.algorithm == "fedpd" else selection_fraction,
            "selection_metric": selection_metric,
            "tie_break_metric": tie_break_metric,
            "search_strategy": "random coarse search followed by focused grid search",
        },
        "best_hyperparameters": best_payload(best_candidate, best_result_row),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def best_payload(best_candidate: Candidate, row: dict[str, Any] | None) -> dict[str, Any]:
    payload = {
        "algorithm_name": best_candidate.algorithm_name,
        "variant": best_candidate.variant,
        "search_stage": best_candidate.search_stage,
        "hyperparameters": best_candidate.hyperparameters,
    }
    if row is not None:
        payload.update(
            {
                "server_accuracy_mean": row["server_accuracy_mean"],
                "server_accuracy_margin_of_error": row["server_accuracy_margin_of_error"],
                "validation_loss_mean": row["validation_loss_mean"],
                "validation_loss_margin_of_error": row["validation_loss_margin_of_error"],
            }
        )
    if best_candidate.group == "FedOpt":
        payload["selected_algorithm"] = best_candidate.algorithm_name
    if best_candidate.group == "FedLT":
        payload["selected_solver"] = best_candidate.hyperparameters["local_solver"]
    return payload


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main tuning workflow
# -----------------------------------------------------------------------------

def load_data() -> tuple[list[Dataset], Dataset, list[str]]:
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
    train_partitions, validation_data = split_train_validation(
        train_dataset.get_partitions(),
        validation_fraction=validation_fraction,
        seed=seed,
    )
    return train_partitions, validation_data, train_dataset.selected_writer_ids


def run_candidate_list(
    candidates: Sequence[Candidate],
    train_partitions: Sequence[Dataset],
    validation_data: Dataset,
    selected_writer_ids: Sequence[str],
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
                selected_writer_ids,
                iterations=config.iterations,
                n_trials=config.n_trials,
                state_snapshot_period=config.iterations,
            )
        except Exception as exc:  # noqa: BLE001
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


def row_to_candidate(row: dict[str, Any], candidates: Sequence[Candidate]) -> Candidate:
    for tuning_candidate in candidates:
        if tuning_candidate.candidate_id == row["candidate_id"]:
            return tuning_candidate
    raise RuntimeError(f"Could not find candidate for row {row['candidate_id']!r}.")


def main() -> None:
    config = parse_args()
    config.run_path.mkdir(parents=True, exist_ok=True)
    candidate_results_path = config.run_path / "exp0_candidate_results.csv"
    best_path = config.run_path / "exp0_best_hyperparameters.json"

    train_partitions, validation_data, selected_writer_ids = load_data()
    metadata = {
        "selected_writer_ids": selected_writer_ids,
        "n_validation_samples": len(validation_data),
        "n_train_samples_after_validation_split": sum(len(partition) for partition in train_partitions),
        "run_config": {
            "algorithm": config.algorithm,
            "iterations": config.iterations,
            "final_iterations": config.final_iterations,
            "n_trials": config.n_trials,
            "n_random_candidates": config.n_random_candidates,
            "max_grid_candidates": config.max_grid_candidates,
            "run_final": config.run_final,
            "selection_fraction": None if config.algorithm == "fedpd" else selection_fraction,
        },
    }
    (config.run_path / "exp0_dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Experiment 0 run path: {config.run_path}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Candidate tuning uses no checkpoint manager. checkpoint_step = None.")

    rng = np.random.default_rng(seed)
    coarse_candidates = random_candidates(config.algorithm, rng, config.n_random_candidates)
    print(f"Running random/coarse search with {len(coarse_candidates)} candidates.")
    all_candidates = list(coarse_candidates)
    all_rows = run_candidate_list(
        coarse_candidates,
        train_partitions,
        validation_data,
        selected_writer_ids,
        config=config,
        candidate_results_path=candidate_results_path,
        starting_index=1,
    )

    if config.algorithm == "fedopt":
        grid_candidates = []
        for fedopt_best_row in best_rows_by_algorithm_name(all_rows):
            fedopt_best_candidate = row_to_candidate(fedopt_best_row, coarse_candidates)
            grid_candidates.extend(
                limit_grid_candidates(
                    grid_candidates_from(fedopt_best_candidate),
                    max_candidates=config.max_grid_candidates,
                    rng=rng,
                )
            )
    else:
        coarse_best_candidate = row_to_candidate(best_row(all_rows), coarse_candidates)
        grid_candidates = limit_grid_candidates(
            grid_candidates_from(coarse_best_candidate),
            max_candidates=config.max_grid_candidates,
            rng=rng,
        )
    print(f"Running focused grid search with {len(grid_candidates)} candidates.")
    all_candidates.extend(grid_candidates)
    grid_rows = run_candidate_list(
        grid_candidates,
        train_partitions,
        validation_data,
        selected_writer_ids,
        config=config,
        candidate_results_path=candidate_results_path,
        starting_index=len(all_rows) + 1,
    )
    all_rows.extend(grid_rows)

    if config.algorithm == "fedlt":
        best_gd_candidate = row_to_candidate(best_row(all_rows), all_candidates)
        solver_candidates = fedlt_solver_candidates(best_gd_candidate)
        print(f"Running FedLT local-solver comparison with {len(solver_candidates)} candidates.")
        all_candidates.extend(solver_candidates)
        solver_rows = run_candidate_list(
            solver_candidates,
            train_partitions,
            validation_data,
            selected_writer_ids,
            config=config,
            candidate_results_path=candidate_results_path,
            starting_index=len(all_rows) + 1,
        )
        all_rows.extend(solver_rows)

    final_best_row = best_row(all_rows)
    final_best_candidate = row_to_candidate(final_best_row, all_candidates)
    save_best_hyperparameters(final_best_candidate, final_best_row, best_path, config=config)

    if config.run_final:
        print(f"Running final best-candidate curve for {config.final_iterations} iterations.")
        run_final_curve(
            final_best_candidate,
            train_partitions,
            validation_data,
            selected_writer_ids,
            config=config,
        )

    print(f"Candidate results saved to: {candidate_results_path}")
    print(f"Best hyperparameters saved to: {best_path}")


if __name__ == "__main__":
    main()
