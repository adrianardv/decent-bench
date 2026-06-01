# ruff: noqa: D103, E402, INP001, T201
"""
FEMNIST aggregation-weighting benchmark under combined communication impairments.

This repeats Experiment 2's uniform-vs-data-size weighted aggregation comparison,
but replaces the clean network with UniformActivationRate(0.5), TopK(0.10), and
UniformDropRate(0.10).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import decent_bench.utils.interoperability as iop
from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.utils import pytorch_initialization
from decent_bench.costs import PyTorchCost
from decent_bench.metrics import metric_library as ml
from decent_bench.networks import FedNetwork
from decent_bench.schemes import NoNoise, TopK, UniformActivationRate, UniformDropRate
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.pytorch_utils import ArgmaxActivation
from experiments.femnist.experiment2 import experiment2_aggregation_weighting as clean_exp2
from experiments.femnist.src import FEMNISTCNN, FEMNISTDatasetHandler

experiment_name = "experiment2_aggregation_weighting_combined_impairments"
condition_key = "combined_uniform_topk_drops"
condition_label = "UniformActivationRate(0.5) + TopK(0.10) + UniformDropRate(0.10)"
metric_result_filename = clean_exp2.metric_result_filename
algorithm_order = clean_exp2.algorithm_order


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


def build_problem() -> tuple[benchmark.BenchmarkProblem, Any, list[str], int, int]:
    iop.set_seed(clean_exp2.seed)
    train_dataset = FEMNISTDatasetHandler(
        split="train",
        n_clients=clean_exp2.n_clients,
        train_fraction=clean_exp2.train_fraction,
        seed=clean_exp2.seed,
        min_train_samples=clean_exp2.min_train_samples,
        min_test_samples=clean_exp2.min_test_samples,
        local_files_only=clean_exp2.local_files_only,
    )
    test_dataset = FEMNISTDatasetHandler(
        split="test",
        n_clients=clean_exp2.n_clients,
        train_fraction=clean_exp2.train_fraction,
        seed=clean_exp2.seed,
        min_train_samples=clean_exp2.min_train_samples,
        min_test_samples=clean_exp2.min_test_samples,
        local_files_only=clean_exp2.local_files_only,
    )

    train_partitions = train_dataset.get_partitions()
    costs = [
        PyTorchCost(
            dataset=partition,
            model=FEMNISTCNN(),
            loss_fn=torch.nn.CrossEntropyLoss(),
            final_activation=ArgmaxActivation(),
            batch_size=min(clean_exp2.batch_size, len(partition)),
            device=clean_exp2.device,
            load_dataset=clean_exp2.load_dataset,
        )
        for partition in train_partitions
    ]
    agents = [
        Agent(
            cost,
            activation=UniformActivationRate(0.50),
            state_snapshot_period=clean_exp2.state_snapshot_period,
            data={"writer_id": writer_id},
        )
        for writer_id, cost in zip(train_dataset.selected_writer_ids, costs, strict=True)
    ]
    network = FedNetwork(
        clients=agents,
        message_noise=NoNoise(),
        message_compression=TopK(0.10),
        message_drop=UniformDropRate(0.10),
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
        "title": "FEMNIST aggregation weighting benchmark under combined impairments",
        "purpose": (
            "Compare uniform and data-size weighted aggregation under simultaneous client availability, "
            "compression, and message-drop impairments."
        ),
        "execution": (
            "one algorithm pair per benchmark() call; each pair writes to "
            "experiment2/combined_uniform_topk_drops/<algorithm>/run_<timestamp>"
        ),
        "dataset": "FEMNIST",
        "dataset_source": "flwrlabs/femnist",
        "partition": "natural writer/client split",
        "n_clients": clean_exp2.n_clients,
        "min_train_samples": clean_exp2.min_train_samples,
        "min_test_samples": clean_exp2.min_test_samples,
        "n_classes": 62,
        "train_fraction": clean_exp2.train_fraction,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "n_trials": clean_exp2.n_trials,
        "iterations": clean_exp2.iterations,
        "state_snapshot_period": clean_exp2.state_snapshot_period,
        "checkpoint_step": clean_exp2.checkpoint_step,
        "batch_size": clean_exp2.batch_size,
        "seed": clean_exp2.seed,
        "device": str(clean_exp2.device),
        "load_dataset": clean_exp2.load_dataset,
        "selected_hyperparameters_path": str(clean_exp2.selected_hyperparameters_path),
        "model": "CNN: conv 1->32, conv 32->64, dense 256, output 62 logits",
        "loss": "torch.nn.CrossEntropyLoss",
        "network": {
            "participation": "partial",
            "client_selection": "UniformSelection",
            "selection_fraction": clean_exp2.selection_fraction,
            "clients_per_round_without_activation_impairment": int(
                clean_exp2.n_clients * clean_exp2.selection_fraction
            ),
            "activation": "UniformActivationRate",
            "activation_probability": 0.50,
            "compression": "TopK",
            "k": 0.10,
            "drops": "UniformDropRate",
            "drop_rate": 0.10,
            "noise": "NoNoise",
            "impairments_apply_bidirectionally": True,
        },
        "aggregation_variants": {
            "uniform": "Each received client upload has equal aggregation weight.",
            "data-size weighted": "Each received client upload is weighted by its local training sample count.",
        },
        "condition": condition_key,
        "condition_label": condition_label,
        "requested_algorithms": requested_algorithms,
        "algorithms": list(algorithm_order),
        "selected_writer_ids": selected_writer_ids,
        "statuses": statuses,
    }
    (run_path / "experiment2_combined_impairments_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def run_algorithm_pair(
    *,
    algorithm_key: str,
    selected_hyperparameters: dict[str, Any],
    run_path: Path,
    selected_writer_ids: list[str] | None,
    n_train_samples: int,
    n_test_samples: int,
    statuses: list[dict[str, Any]],
) -> tuple[list[str], int, int]:
    print(f"Running {algorithm_key} under {condition_key}; results: {run_path}")

    problem, x0, writer_ids, current_n_train_samples, current_n_test_samples = build_problem()
    if selected_writer_ids is None:
        selected_writer_ids = writer_ids
        n_train_samples = current_n_train_samples
        n_test_samples = current_n_test_samples
    elif writer_ids != selected_writer_ids:
        raise RuntimeError("Selected writer IDs changed between algorithm runs.")

    algorithms = clean_exp2.build_algorithms(algorithm_key, x0, selected_hyperparameters)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=run_path,
        checkpoint_step=clean_exp2.checkpoint_step,
        keep_n_checkpoints=1,
        benchmark_metadata={
            "experiment": experiment_name,
            "condition": condition_key,
            "algorithm": algorithm_key,
            "aggregation_variants": [algorithm.name for algorithm in algorithms],
            "n_trials": clean_exp2.n_trials,
            "iterations": clean_exp2.iterations,
            "state_snapshot_period": clean_exp2.state_snapshot_period,
            "checkpoint_step": clean_exp2.checkpoint_step,
        },
    )

    try:
        result = benchmark.benchmark(
            algorithms=algorithms,
            benchmark_problem=problem,
            n_trials=clean_exp2.n_trials,
            max_processes=1,
            progress_step=clean_exp2.progress_step,
            show_speed=True,
            show_trial=True,
            checkpoint_manager=checkpoint_manager,
            log_level=logging.INFO,
        )
        metric_result = benchmark.compute_metrics(
            benchmark_result=result,
            table_metrics=build_table_metrics(),
            plot_metrics=clean_exp2.build_plot_metrics(),
            checkpoint_manager=checkpoint_manager,
            log_level=logging.INFO,
        )
        benchmark.display_metrics(
            metrics_result=metric_result,
            checkpoint_manager=checkpoint_manager,
            individual_plots=True,
            show_plots=clean_exp2.show_plots,
            log_level=logging.INFO,
        )
        clean_exp2.save_metric_dataframes(metric_result, checkpoint_manager.get_results_path())
        metric_result.agent_metrics = None
        clean_exp2.save_pickle_zst(metric_result, run_path / metric_result_filename)
        (run_path / "metric_computation_complete.json").write_text(
            json.dumps({"metric_computation_complete": True}, indent=2),
            encoding="utf-8",
        )
        statuses.append(
            {
                "algorithm": algorithm_key,
                "condition": condition_key,
                "status": "ok",
                "variants": [algorithm.name for algorithm in algorithms],
            }
        )

    except Exception as error:
        statuses.append(
            {
                "algorithm": algorithm_key,
                "condition": condition_key,
                "status": "failed",
                "variants": [algorithm.name for algorithm in algorithms],
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
        clean_exp2.clear_cuda_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        default="all",
        choices=["all", *algorithm_order],
        help="Federated algorithm to run, or 'all' to run all algorithm pairs sequentially.",
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
        run_id = f"{run_id}_{clean_exp2.slugify(args.run_label)}"
    checkpoint_root = Path("experiments/femnist/checkpoints/experiment2") / condition_key

    selected_hyperparameters = clean_exp2.load_selected_hyperparameters()
    selected_writer_ids: list[str] | None = None
    n_train_samples = 0
    n_test_samples = 0

    print(f"Writing combined-impairment Experiment 2 results under: {checkpoint_root}")

    for algorithm_key in requested_algorithms:
        run_path = checkpoint_root / algorithm_key / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        statuses: list[dict[str, Any]] = []

        try:
            selected_writer_ids, n_train_samples, n_test_samples = run_algorithm_pair(
                algorithm_key=algorithm_key,
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
                requested_algorithms=[algorithm_key],
                statuses=statuses,
            )

    print(f"Combined-impairment Experiment 2 benchmark complete: {checkpoint_root}")


if __name__ == "__main__":
    main()
