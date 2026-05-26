from __future__ import annotations

import csv
import gc
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
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
from decent_bench.schemes import AlwaysActive, NoCompression, NoDrops, NoNoise
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.pytorch_utils import ArgmaxActivation
from decent_bench.utils.types import Dataset, SupportedDevices

from experiments.femnist.src import FEMNISTCNN, FEMNISTDatasetHandler


run_path = Path("experiments/femnist/checkpoints/experiment0") / f"run_{datetime.now():%Y%m%d_%H%M%S}"

seed = 20260524
n_clients = 100
min_train_samples = 100
min_test_samples = 20
train_fraction = 0.8
validation_fraction = 0.2
n_trials = 2
iterations = 400
state_snapshot_period = iterations // 2
progress_step = max(1, iterations // 10)
checkpoint_step = None
batch_size = 32
device = SupportedDevices.GPU
local_files_only = False
load_dataset = True
show_plots = False

selection_metric = "server accuracy"
tie_break_metric = "loss"

candidate_result_fields = [
    "status",
    "candidate_id",
    "algorithm_group",
    "algorithm_name",
    "variant",
    "server_accuracy_mean",
    "server_accuracy_margin_of_error",
    "validation_loss_mean",
    "validation_loss_margin_of_error",
    "checkpoint_dir",
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
class Candidate:
    group: str
    algorithm_name: str
    variant: str
    hyperparameters: dict[str, Any]
    algorithm_factory: Callable[[dict[str, Any], Any], Any]

    @property
    def candidate_id(self) -> str:
        parts = [self.group, self.algorithm_name, self.variant]
        compact = "_".join(part for part in parts if part)
        return "".join(char if char.isalnum() or char in "-_" else "_" for char in compact)


def fedavg_factory(params: dict[str, Any], x0: Any) -> FedAvg:
    return FedAvg(iterations=iterations, selection_scheme=None, x0=x0, **params)


def fedprox_factory(params: dict[str, Any], x0: Any) -> FedProx:
    return FedProx(iterations=iterations, selection_scheme=None, x0=x0, **params)


def scaffold_factory(params: dict[str, Any], x0: Any) -> Scaffold:
    return Scaffold(iterations=iterations, selection_scheme=None, x0=x0, **params)


def fednova_factory(params: dict[str, Any], x0: Any) -> FedNova:
    params = dict(params)
    params["num_local_steps"] = params.pop("num_local_epochs")
    return FedNova(iterations=iterations, selection_scheme=None, x0=x0, **params)


def fedadam_factory(params: dict[str, Any], x0: Any) -> FedAdam:
    return FedAdam(iterations=iterations, selection_scheme=None, x0=x0, **params)


def fedyogi_factory(params: dict[str, Any], x0: Any) -> FedYogi:
    return FedYogi(iterations=iterations, selection_scheme=None, x0=x0, **params)


def fedadagrad_factory(params: dict[str, Any], x0: Any) -> FedAdagrad:
    return FedAdagrad(iterations=iterations, selection_scheme=None, x0=x0, **params)


def fedlt_factory(params: dict[str, Any], x0: Any) -> FedLT:
    return FedLT(iterations=iterations, selection_scheme=None, x0=x0, **params)


def feddyn_factory(params: dict[str, Any], x0: Any) -> FedDyn:
    return FedDyn(iterations=iterations, selection_scheme=None, x0=x0, **params)


def fedpd_factory(params: dict[str, Any], x0: Any) -> FedPD:
    params = dict(params)
    params["num_local_steps"] = params.pop("num_local_epochs")
    return FedPD(iterations=iterations, x0=x0, **params)


def candidate(
    group: str,
    algorithm_name: str,
    variant: str,
    hyperparameters: dict[str, Any],
    algorithm_factory: Callable[[dict[str, Any], Any], Any],
) -> Candidate:
    return Candidate(
        group=group,
        algorithm_name=algorithm_name,
        variant=variant,
        hyperparameters=hyperparameters,
        algorithm_factory=algorithm_factory,
    )


def tuning_candidates() -> list[Candidate]:
    return [
        candidate("FedAvg", "FedAvg", "lr_0.005_e1", {"step_size": 0.005, "num_local_epochs": 1}, fedavg_factory),
        candidate("FedAvg", "FedAvg", "lr_0.01_e1", {"step_size": 0.01, "num_local_epochs": 1}, fedavg_factory),
        candidate("FedAvg", "FedAvg", "lr_0.01_e3", {"step_size": 0.01, "num_local_epochs": 3}, fedavg_factory),
        candidate("FedAvg", "FedAvg", "lr_0.02_e3", {"step_size": 0.02, "num_local_epochs": 3}, fedavg_factory),
        candidate(
            "FedProx",
            "FedProx",
            "lr_0.005_e1_mu_0.01",
            {"step_size": 0.005, "num_local_epochs": 1, "mu": 0.01},
            fedprox_factory,
        ),
        candidate(
            "FedProx",
            "FedProx",
            "lr_0.01_e1_mu_0.01",
            {"step_size": 0.01, "num_local_epochs": 1, "mu": 0.01},
            fedprox_factory,
        ),
        candidate(
            "FedProx",
            "FedProx",
            "lr_0.01_e3_mu_0.001",
            {"step_size": 0.01, "num_local_epochs": 3, "mu": 0.001},
            fedprox_factory,
        ),
        candidate(
            "FedProx",
            "FedProx",
            "lr_0.01_e3_mu_0.1",
            {"step_size": 0.01, "num_local_epochs": 3, "mu": 0.1},
            fedprox_factory,
        ),
        candidate(
            "SCAFFOLD",
            "Scaffold",
            "lr_0.005_e1_server_1",
            {"step_size": 0.005, "num_local_epochs": 1, "server_step_size": 1.0},
            scaffold_factory,
        ),
        candidate(
            "SCAFFOLD",
            "Scaffold",
            "lr_0.01_e1_server_1",
            {"step_size": 0.01, "num_local_epochs": 1, "server_step_size": 1.0},
            scaffold_factory,
        ),
        candidate(
            "SCAFFOLD",
            "Scaffold",
            "lr_0.01_e3_server_0.5",
            {"step_size": 0.01, "num_local_epochs": 3, "server_step_size": 0.5},
            scaffold_factory,
        ),
        candidate(
            "SCAFFOLD",
            "Scaffold",
            "lr_0.01_e3_server_1",
            {"step_size": 0.01, "num_local_epochs": 3, "server_step_size": 1.0},
            scaffold_factory,
        ),
        candidate(
            "FedNova",
            "FedNova",
            "plain_lr_0.01_e3",
            {"step_size": 0.01, "num_local_epochs": 3},
            fednova_factory,
        ),
        candidate(
            "FedNova",
            "FedNova",
            "momentum_beta_0.9",
            {"step_size": 0.01, "num_local_epochs": 3, "use_momentum": True, "beta": 0.9},
            fednova_factory,
        ),
        candidate(
            "FedNova",
            "FedNova",
            "prox_mu_0.01",
            {"step_size": 0.01, "num_local_epochs": 3, "use_prox": True, "mu": 0.01},
            fednova_factory,
        ),
        candidate(
            "FedNova",
            "FedNova",
            "server_momentum_gamma_0.9",
            {"step_size": 0.01, "num_local_epochs": 3, "use_server_momentum": True, "gamma": 0.9},
            fednova_factory,
        ),
        *fedopt_candidates(),
        *fedlt_candidates(),
        candidate(
            "FedDyn",
            "FedDyn",
            "alpha_0.001",
            {"step_size": 0.01, "num_local_epochs": 3, "alpha": 0.001},
            feddyn_factory,
        ),
        candidate(
            "FedDyn",
            "FedDyn",
            "alpha_0.01",
            {"step_size": 0.01, "num_local_epochs": 3, "alpha": 0.01},
            feddyn_factory,
        ),
        candidate(
            "FedDyn",
            "FedDyn",
            "alpha_0.1",
            {"step_size": 0.005, "num_local_epochs": 3, "alpha": 0.1},
            feddyn_factory,
        ),
        candidate(
            "FedPD",
            "FedPD",
            "eta_0.1_skip_0",
            {"step_size": 0.005, "num_local_epochs": 3, "eta": 0.1, "skip_probability": 0.0},
            fedpd_factory,
        ),
        candidate(
            "FedPD",
            "FedPD",
            "eta_1_skip_0",
            {"step_size": 0.005, "num_local_epochs": 3, "eta": 1.0, "skip_probability": 0.0},
            fedpd_factory,
        ),
        candidate(
            "FedPD",
            "FedPD",
            "eta_1_skip_0.1",
            {"step_size": 0.005, "num_local_epochs": 3, "eta": 1.0, "skip_probability": 0.1},
            fedpd_factory,
        ),
    ]


def fedopt_candidates() -> list[Candidate]:
    variants = [
        ("FedAdam", fedadam_factory, {"beta_2": 0.99}),
        ("FedYogi", fedyogi_factory, {"beta_2": 0.99}),
        ("FedAdagrad", fedadagrad_factory, {}),
    ]
    shared = [
        ("lr_0.005_e1_server_0.001", {"step_size": 0.005, "num_local_epochs": 1, "server_step_size": 0.001}),
        ("lr_0.01_e1_server_0.001", {"step_size": 0.01, "num_local_epochs": 1, "server_step_size": 0.001}),
        ("lr_0.01_e3_server_0.001", {"step_size": 0.01, "num_local_epochs": 3, "server_step_size": 0.001}),
        ("lr_0.01_e3_server_0.01", {"step_size": 0.01, "num_local_epochs": 3, "server_step_size": 0.01}),
    ]
    candidates: list[Candidate] = []
    for algorithm_name, factory, extra in variants:
        for variant_name, params in shared:
            params = {
                **params,
                "beta_1": 0.9,
                "tau": 1e-6,
                **extra,
            }
            candidates.append(candidate("FedOpt", algorithm_name, variant_name, params, factory))
    return candidates


def fedlt_candidates() -> list[Candidate]:
    return [
        candidate(
            "FedLT",
            "FedLT",
            "gd_lr_0.005_e3_rho_1",
            {"step_size": 0.005, "num_local_epochs": 3, "rho": 1.0, "local_solver": "gd"},
            fedlt_factory,
        ),
        candidate(
            "FedLT",
            "FedLT",
            "gd_lr_0.01_e3_rho_1",
            {"step_size": 0.01, "num_local_epochs": 3, "rho": 1.0, "local_solver": "gd"},
            fedlt_factory,
        ),
        candidate(
            "FedLT",
            "FedLT",
            "gd_lr_0.01_e3_rho_0.1",
            {"step_size": 0.01, "num_local_epochs": 3, "rho": 0.1, "local_solver": "gd"},
            fedlt_factory,
        ),
        candidate(
            "FedLT",
            "FedLT",
            "adam_lr_0.005_e3_rho_1",
            {
                "step_size": 0.005,
                "num_local_epochs": 3,
                "rho": 1.0,
                "local_solver": "adam",
                "solver_args": {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
            },
            fedlt_factory,
        ),
        candidate(
            "FedLT",
            "FedLT",
            "nesterov_lr_0.005_e3_rho_1",
            {
                "step_size": 0.005,
                "num_local_epochs": 3,
                "rho": 1.0,
                "local_solver": "nesterov",
                "solver_args": {"momentum": 0.9},
            },
            fedlt_factory,
        ),
    ]


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
        n_validation = max(1, int(round(len(partition) * validation_fraction)))
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


def run_candidate(
    tuning_candidate: Candidate,
    train_partitions: Sequence[Dataset],
    validation_data: Dataset,
    selected_writer_ids: Sequence[str],
    candidate_index: int,
) -> dict[str, Any]:
    candidate_path = run_path / "candidate_checkpoints" / f"{candidate_index:03d}_{tuning_candidate.candidate_id}"
    problem, x0 = build_problem(train_partitions, validation_data, selected_writer_ids)
    algorithm = tuning_candidate.algorithm_factory(tuning_candidate.hyperparameters, x0)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=candidate_path,
        checkpoint_step=checkpoint_step,
        keep_n_checkpoints=1,
        benchmark_metadata={
            "experiment": "experiment0",
            "candidate_id": tuning_candidate.candidate_id,
            "algorithm_group": tuning_candidate.group,
            "algorithm_name": tuning_candidate.algorithm_name,
            "variant": tuning_candidate.variant,
            "hyperparameters": tuning_candidate.hyperparameters,
            "selection_metric": selection_metric,
            "tie_break_metric": tie_break_metric,
            "dataset": "FEMNIST",
            "n_clients": n_clients,
            "n_trials": n_trials,
            "iterations": iterations,
            "seed": seed,
            "validation_fraction": validation_fraction,
        },
    )
    result = benchmark.benchmark(
        algorithms=[algorithm],
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
        checkpoint_manager=checkpoint_manager,
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
        "candidate_id": tuning_candidate.candidate_id,
        "algorithm_group": tuning_candidate.group,
        "algorithm_name": tuning_candidate.algorithm_name,
        "variant": tuning_candidate.variant,
        "server_accuracy_mean": server_accuracy,
        "server_accuracy_margin_of_error": server_accuracy_ci,
        "validation_loss_mean": validation_loss,
        "validation_loss_margin_of_error": validation_loss_ci,
        "checkpoint_dir": str(candidate_path),
        **flatten_hyperparameters(tuning_candidate.hyperparameters),
    }


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


def update_best(best_by_group: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    if row["status"] != "ok":
        return
    current = best_by_group.get(row["algorithm_group"])
    if current is None or is_better(row, current):
        best_by_group[row["algorithm_group"]] = row


def is_better(candidate_row: dict[str, Any], current_row: dict[str, Any]) -> bool:
    candidate_accuracy = candidate_row["server_accuracy_mean"]
    current_accuracy = current_row["server_accuracy_mean"]
    if candidate_accuracy != current_accuracy:
        return candidate_accuracy > current_accuracy
    return candidate_row["validation_loss_mean"] < current_row["validation_loss_mean"]


def save_best_hyperparameters(best_by_group: dict[str, dict[str, Any]], path: Path) -> None:
    payload = {
        "metadata": {
            "experiment": "experiment0",
            "dataset": "FEMNIST",
            "dataset_source": "flwrlabs/femnist",
            "partition": "natural writer/client split",
            "n_clients": n_clients,
            "min_train_samples": min_train_samples,
            "min_test_samples": min_test_samples,
            "train_fraction": train_fraction,
            "validation_fraction_from_train": validation_fraction,
            "n_trials": n_trials,
            "iterations": iterations,
            "state_snapshot_period": state_snapshot_period,
            "checkpoint_step": checkpoint_step,
            "batch_size": batch_size,
            "seed": seed,
            "selection_metric": selection_metric,
            "tie_break_metric": tie_break_metric,
        },
        "best_hyperparameters": {group: best_payload(row) for group, row in best_by_group.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def best_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "algorithm_name": row["algorithm_name"],
        "variant": row["variant"],
        "server_accuracy_mean": row["server_accuracy_mean"],
        "server_accuracy_margin_of_error": row["server_accuracy_margin_of_error"],
        "validation_loss_mean": row["validation_loss_mean"],
        "validation_loss_margin_of_error": row["validation_loss_margin_of_error"],
        "checkpoint_dir": row["checkpoint_dir"],
        "hyperparameters": {
            key: row[key]
            for key in candidate_result_fields
            if key in row
            and key
            not in {
                "status",
                "candidate_id",
                "algorithm_group",
                "algorithm_name",
                "variant",
                "server_accuracy_mean",
                "server_accuracy_margin_of_error",
                "validation_loss_mean",
                "validation_loss_margin_of_error",
                "checkpoint_dir",
                "error",
            }
        },
    }
    if row["algorithm_group"] == "FedOpt":
        payload["selected_algorithm"] = row["algorithm_name"]
    if row["algorithm_group"] == "FedLT" and "local_solver" in row:
        payload["selected_solver"] = row["local_solver"]
    return payload


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    run_path.mkdir(parents=True, exist_ok=True)
    candidate_results_path = run_path / "exp0_candidate_results.csv"
    best_path = run_path / "exp0_best_hyperparameters.json"

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
    selected_writer_ids = train_dataset.selected_writer_ids

    metadata = {
        "selected_writer_ids": selected_writer_ids,
        "n_validation_samples": len(validation_data),
        "n_train_samples_after_validation_split": sum(len(partition) for partition in train_partitions),
    }
    (run_path / "exp0_dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_by_group: dict[str, dict[str, Any]] = {}
    candidates = tuning_candidates()
    print(f"Experiment 0 run path: {run_path}")
    print(f"Running {len(candidates)} candidates sequentially.")

    for candidate_index, tuning_candidate in enumerate(candidates, start=1):
        print(
            f"[{candidate_index}/{len(candidates)}] "
            f"{tuning_candidate.group} - {tuning_candidate.algorithm_name} - {tuning_candidate.variant}"
        )
        try:
            row = run_candidate(
                tuning_candidate,
                train_partitions,
                validation_data,
                selected_writer_ids,
                candidate_index,
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("Candidate failed: %s", tuning_candidate.candidate_id)
            row = {
                "status": "failed",
                "candidate_id": tuning_candidate.candidate_id,
                "algorithm_group": tuning_candidate.group,
                "algorithm_name": tuning_candidate.algorithm_name,
                "variant": tuning_candidate.variant,
                "error": repr(exc),
                **flatten_hyperparameters(tuning_candidate.hyperparameters),
            }
        append_candidate_result(row, candidate_results_path)
        update_best(best_by_group, row)
        save_best_hyperparameters(best_by_group, best_path)
        cleanup_cuda()

    print(f"Candidate results saved to: {candidate_results_path}")
    print(f"Best hyperparameters saved to: {best_path}")


if __name__ == "__main__":
    main()
