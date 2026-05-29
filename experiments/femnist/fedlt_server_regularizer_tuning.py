"""Focused FedLT tuning with an L2 server regularizer for FEMNIST."""

from __future__ import annotations

import csv
import gc
import json
import logging
import math
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import decent_bench.utils.interoperability as iop
import numpy as np
import torch
import zstandard as zstd
from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedLT
from decent_bench.algorithms.utils import pytorch_initialization
from decent_bench.costs import L2RegularizerCost, PyTorchCost
from decent_bench.metrics import metric_library as ml
from decent_bench.networks import FedNetwork
from decent_bench.schemes import AlwaysActive, NoCompression, NoDrops, NoNoise, UniformSelection
from decent_bench.utils.pytorch_utils import ArgmaxActivation
from decent_bench.utils.types import Dataset, SupportedDevices

from experiments.femnist.src import FEMNISTCNN, FEMNISTDatasetHandler


SEED = 20260524
N_CLIENTS = 100
MIN_TRAIN_SAMPLES = 100
MIN_TEST_SAMPLES = 20
TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 32
DEVICE = SupportedDevices.GPU
LOCAL_FILES_ONLY = False
LOAD_DATASET = True

ITERATIONS = 1000
FINAL_ITERATIONS = 2000
N_TRIALS = 1
SELECTION_FRACTION = 0.2
N_RANDOM_CANDIDATES = 18

RUN_PATH = (
    Path("experiments/femnist/checkpoints/experiment0/fedlt_server_regularizer")
    / f"run_{datetime.now():%Y%m%d_%H%M%S}"
)
METRIC_RESULT_FILENAME = "metric_computation.pkl.zst"

CANDIDATE_FIELDS = [
    "status",
    "candidate_id",
    "search_stage",
    "server_accuracy_mean",
    "server_accuracy_margin_of_error",
    "validation_loss_mean",
    "validation_loss_margin_of_error",
    "elapsed_seconds",
    "error",
    "step_size",
    "num_local_epochs",
    "rho",
    "local_solver",
    "server_l2_weight",
    "solver_args.beta1",
    "solver_args.beta2",
    "solver_args.epsilon",
    "solver_args.momentum",
]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    search_stage: str
    hyperparameters: dict[str, Any]


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def random_choice(rng: np.random.Generator, values: list[Any]) -> Any:
    return values[int(rng.integers(0, len(values)))]


def build_candidates() -> list[Candidate]:
    rng = np.random.default_rng(SEED)
    candidates: list[Candidate] = []

    candidates.append(
        Candidate(
            candidate_id="reference_adam_l2_1e_m3_lr_0p01_e5_rho_1",
            search_stage="reference",
            hyperparameters={
                "step_size": 0.01,
                "num_local_epochs": 5,
                "rho": 1.0,
                "local_solver": "adam",
                "server_l2_weight": 1e-3,
                "solver_args": {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
            },
        )
    )
    candidates.append(
        Candidate(
            candidate_id="reference_gd_l2_1e_m3_lr_0p0244_e3_rho_1",
            search_stage="reference",
            hyperparameters={
                "step_size": 0.02441691061516309,
                "num_local_epochs": 3,
                "rho": 1.0,
                "local_solver": "gd",
                "server_l2_weight": 1e-3,
            },
        )
    )

    for index in range(N_RANDOM_CANDIDATES):
        solver = random_choice(rng, ["adam", "gd"])
        params: dict[str, Any] = {
            "step_size": log_uniform(rng, 0.003, 0.03),
            "num_local_epochs": random_choice(rng, [3, 5, 8]),
            "rho": random_choice(rng, [0.1, 1.0]),
            "local_solver": solver,
            "server_l2_weight": log_uniform(rng, 1e-5, 1e-2),
        }
        if solver == "adam":
            params["solver_args"] = {
                "beta1": random_choice(rng, [0.5, 0.9]),
                "beta2": random_choice(rng, [0.99, 0.999]),
                "epsilon": 1e-8,
            }

        candidates.append(
            Candidate(
                candidate_id=f"random_{index:02d}_{solver}",
                search_stage="random",
                hyperparameters=params,
            )
        )

    return candidates


def split_train_validation(
    train_partitions: list[Dataset],
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
        validation_data.extend(
            datapoint for index, datapoint in enumerate(partition) if index in validation_indices
        )

    return tuning_train_partitions, validation_data


def load_data() -> tuple[list[Dataset], Dataset, list[str]]:
    iop.set_seed(SEED)
    train_dataset = FEMNISTDatasetHandler(
        split="train",
        n_clients=N_CLIENTS,
        train_fraction=TRAIN_FRACTION,
        seed=SEED,
        min_train_samples=MIN_TRAIN_SAMPLES,
        min_test_samples=MIN_TEST_SAMPLES,
        local_files_only=LOCAL_FILES_ONLY,
    )
    train_partitions, validation_data = split_train_validation(
        list(train_dataset.get_partitions()),
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )
    return train_partitions, validation_data, train_dataset.selected_writer_ids


def build_problem(
    train_partitions: list[Dataset],
    validation_data: Dataset,
    selected_writer_ids: list[str],
    *,
    server_l2_weight: float,
    state_snapshot_period: int,
) -> tuple[benchmark.BenchmarkProblem, Any]:
    iop.set_seed(SEED)

    costs = [
        PyTorchCost(
            dataset=partition,
            model=FEMNISTCNN(),
            loss_fn=torch.nn.CrossEntropyLoss(),
            final_activation=ArgmaxActivation(),
            batch_size=min(BATCH_SIZE, len(partition)),
            device=DEVICE,
            load_dataset=LOAD_DATASET,
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

    base_cost = costs[0]
    server_cost = L2RegularizerCost(
        base_cost.shape,
        framework=base_cost.framework,
        device=base_cost.device,
    ) * server_l2_weight
    server = Agent(
        server_cost,
        activation=AlwaysActive(),
        state_snapshot_period=state_snapshot_period,
        data={"role": "server", "server_l2_weight": server_l2_weight},
    )

    network = FedNetwork(
        clients=agents,
        server=server,
        message_noise=NoNoise(),
        message_compression=NoCompression(),
        message_drop=NoDrops(),
    )
    problem = benchmark.BenchmarkProblem(network=network, test_data=validation_data)
    x0 = pytorch_initialization(network, all_same=True)
    return problem, x0


def flatten_hyperparameters(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in hyperparameters.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat[f"{key}.{nested_key}"] = nested_value
        else:
            flat[key] = value
    return flat


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


def append_candidate_result(row: dict[str, Any]) -> None:
    csv_path = RUN_PATH / "exp0_candidate_results.csv"
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_algorithm(candidate: Candidate, x0: Any, iterations: int) -> FedLT:
    params = dict(candidate.hyperparameters)
    server_l2_weight = params.pop("server_l2_weight")
    algorithm = FedLT(
        iterations=iterations,
        selection_scheme=UniformSelection(fraction_selected_clients=SELECTION_FRACTION),
        x0=x0,
        **params,
    )
    algorithm.name = f"FedLT-L2Server-{server_l2_weight:.1e}"
    return algorithm


def run_candidate(
    candidate: Candidate,
    train_partitions: list[Dataset],
    validation_data: Dataset,
    selected_writer_ids: list[str],
) -> dict[str, Any]:
    start_time = time.perf_counter()
    server_l2_weight = float(candidate.hyperparameters["server_l2_weight"])
    problem, x0 = build_problem(
        train_partitions,
        validation_data,
        selected_writer_ids,
        server_l2_weight=server_l2_weight,
        state_snapshot_period=ITERATIONS,
    )
    algorithm = build_algorithm(candidate, x0, ITERATIONS)

    result = benchmark.benchmark(
        algorithms=[algorithm],
        benchmark_problem=problem,
        n_trials=N_TRIALS,
        max_processes=1,
        progress_step=max(1, ITERATIONS // 10),
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
        "candidate_id": candidate.candidate_id,
        "search_stage": candidate.search_stage,
        "server_accuracy_mean": server_accuracy,
        "server_accuracy_margin_of_error": server_accuracy_ci,
        "validation_loss_mean": validation_loss,
        "validation_loss_margin_of_error": validation_loss_ci,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        **flatten_hyperparameters(candidate.hyperparameters),
    }


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def finite_loss(row: dict[str, Any]) -> float:
    value = float(row["validation_loss_mean"])
    return value if math.isfinite(value) else float("inf")


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    finite_rows = [row for row in ok_rows if math.isfinite(float(row["validation_loss_mean"]))]
    source_rows = finite_rows or ok_rows
    if not source_rows:
        raise RuntimeError("No successful candidates were available.")
    return max(source_rows, key=lambda row: (float(row["server_accuracy_mean"]), -finite_loss(row)))


def candidate_by_id(candidates: list[Candidate], candidate_id: str) -> Candidate:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise RuntimeError(f"Could not find candidate {candidate_id!r}")


def best_payload(candidate: Candidate, row: dict[str, Any] | None) -> dict[str, Any]:
    payload = {
        "algorithm_name": "FedLT",
        "variant": candidate.candidate_id,
        "search_stage": candidate.search_stage,
        "hyperparameters": candidate.hyperparameters,
        "selected_solver": candidate.hyperparameters["local_solver"],
        "server_regularizer": "scaled L2",
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
    return payload


def save_best(best_candidate: Candidate, best_row: dict[str, Any]) -> None:
    payload = {
        "metadata": {
            "experiment": "experiment0",
            "algorithm": "fedlt_server_regularizer",
            "dataset": "FEMNIST",
            "dataset_source": "flwrlabs/femnist",
            "partition": "natural writer/client split",
            "n_clients": N_CLIENTS,
            "min_train_samples": MIN_TRAIN_SAMPLES,
            "min_test_samples": MIN_TEST_SAMPLES,
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction_from_train": VALIDATION_FRACTION,
            "n_trials": N_TRIALS,
            "iterations": ITERATIONS,
            "state_snapshot_period": ITERATIONS,
            "checkpoint_step": None,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "selection_fraction": SELECTION_FRACTION,
            "selection_metric": "server accuracy",
            "tie_break_metric": "loss",
            "search_strategy": "small random search around FedLT with scaled L2 server regularizer",
        },
        "best_hyperparameters": best_payload(best_candidate, best_row),
    }
    (RUN_PATH / "exp0_best_hyperparameters.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_pickle_zst(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor(level=1)
    with path.open("wb") as file_obj, compressor.stream_writer(file_obj) as compressed_writer:
        pickle.dump(data, compressed_writer, protocol=pickle.HIGHEST_PROTOCOL)


def run_final_curve(
    best_candidate: Candidate,
    train_partitions: list[Dataset],
    validation_data: Dataset,
    selected_writer_ids: list[str],
) -> None:
    final_path = RUN_PATH / "final_best_candidate_curve"
    final_path.mkdir(parents=True, exist_ok=True)

    state_snapshot_period = max(1, FINAL_ITERATIONS // 10)
    problem, x0 = build_problem(
        train_partitions,
        validation_data,
        selected_writer_ids,
        server_l2_weight=float(best_candidate.hyperparameters["server_l2_weight"]),
        state_snapshot_period=state_snapshot_period,
    )
    algorithm = build_algorithm(best_candidate, x0, FINAL_ITERATIONS)

    result = benchmark.benchmark(
        algorithms=[algorithm],
        benchmark_problem=problem,
        n_trials=N_TRIALS,
        max_processes=1,
        progress_step=max(1, FINAL_ITERATIONS // 10),
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

    metric_result.agent_metrics = None
    save_pickle_zst(metric_result, final_path / METRIC_RESULT_FILENAME)
    (final_path / "metric_computation_complete.json").write_text(
        json.dumps({"metric_computation_complete": True}, indent=2),
        encoding="utf-8",
    )
    (final_path / "metadata.json").write_text(
        json.dumps(
            {
                "final_iterations": FINAL_ITERATIONS,
                "state_snapshot_period": state_snapshot_period,
                "n_trials": N_TRIALS,
                "best_candidate": best_payload(best_candidate, row=None),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_dataset_metadata(selected_writer_ids: list[str], validation_data: Dataset, train_partitions: list[Dataset]) -> None:
    payload = {
        "selected_writer_ids": selected_writer_ids,
        "n_validation_samples": len(validation_data),
        "n_train_samples_after_validation_split": sum(len(partition) for partition in train_partitions),
        "run_config": {
            "algorithm": "fedlt_server_regularizer",
            "iterations": ITERATIONS,
            "final_iterations": FINAL_ITERATIONS,
            "n_trials": N_TRIALS,
            "run_final": True,
            "selection_fraction": SELECTION_FRACTION,
            "candidate_count": 2 + N_RANDOM_CANDIDATES,
            "server_regularizer": "scaled L2",
        },
        "notes": (
            "Standalone diagnostic FedLT run with a scaled L2 server regularizer. "
            "This changes the FedLT server proximal step relative to the default ZeroCost server."
        ),
    }
    (RUN_PATH / "exp0_dataset_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    RUN_PATH.mkdir(parents=True, exist_ok=True)
    train_partitions, validation_data, selected_writer_ids = load_data()
    save_dataset_metadata(selected_writer_ids, validation_data, train_partitions)

    candidates = build_candidates()
    rows: list[dict[str, Any]] = []

    print(f"Running {len(candidates)} FedLT server-regularizer candidates")
    print(f"Results path: {RUN_PATH}")

    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}] {candidate.candidate_id}")
        try:
            row = run_candidate(candidate, train_partitions, validation_data, selected_writer_ids)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Candidate failed: %s", candidate.candidate_id)
            row = {
                "status": "failed",
                "candidate_id": candidate.candidate_id,
                "search_stage": candidate.search_stage,
                "error": repr(exc),
                **flatten_hyperparameters(candidate.hyperparameters),
            }
        rows.append(row)
        append_candidate_result(row)
        cleanup_cuda()

    selected_row = choose_best(rows)
    selected_candidate = candidate_by_id(candidates, selected_row["candidate_id"])
    save_best(selected_candidate, selected_row)
    run_final_curve(selected_candidate, train_partitions, validation_data, selected_writer_ids)

    print(f"FedLT server-regularizer tuning complete: {RUN_PATH}")


if __name__ == "__main__":
    main()
