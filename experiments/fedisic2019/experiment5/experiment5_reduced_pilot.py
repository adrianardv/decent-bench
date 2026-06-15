# ruff: noqa: ANN401, D103, E402, INP001, PLR0911, T201
"""
Fed-ISIC2019 reduced-budget communication-impairment pilot.

This pilot is for deadline-driven diagnostics when the full Experiment 5 is
too expensive. It uses deterministic stratified per-center proportional
sampling and should not be interpreted as the official full-data benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import decent_bench.utils.interoperability as iop
from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.algorithms.federated import FedAdagrad, FedAdam, FedAvg, FedLT, FedPD, FedYogi, Scaffold
from decent_bench.algorithms.utils import pytorch_initialization
from decent_bench.costs import PyTorchCost
from decent_bench.networks import FedNetwork
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.pytorch_utils import ArgmaxActivation
from decent_bench.utils.types import Datapoint, SupportedDevices
from experiments.fedisic2019.experiment5 import experiment5_communication_impairments as full_exp5
from experiments.fedisic2019.src import WeightedFocalLoss, build_model, build_reduced_handlers, write_reduced_distribution_outputs


seed = 20260524
n_trials = 2
iterations = 1000
state_snapshot_period = 200
progress_step = 100
checkpoint_step = None
batch_size = 64
device = SupportedDevices.GPU
local_files_only = False
load_dataset = False
show_plots = False
model_name = "efficientnet_b0"
pretrained = True

sample_fraction_per_center = 0.10
min_samples_per_center = 100

experiment_group = "experiment5_reduced_pilot"
experiment_name = "experiment5_reduced_pilot"
checkpoint_root = Path("experiments/fedisic2019/checkpoints") / experiment_group
metric_result_filename = "metric_computation.pkl.zst"

condition_keys = ("clean_baseline", "combination")
algorithm_order = ("fedavg", "fedlt", "fedopt", "fedpd", "scaffold")


def condition_lookup() -> dict[str, full_exp5.Condition]:
    return {key: full_exp5.condition_lookup()[key] for key in condition_keys}


def build_reduced_problem(
    condition: full_exp5.Condition,
    *,
    run_path: Path,
) -> tuple[benchmark.BenchmarkProblem, Any, dict[str, Any]]:
    iop.set_seed(seed)
    train_dataset, test_dataset = build_reduced_handlers(
        sample_fraction_per_center=sample_fraction_per_center,
        min_samples_per_center=min_samples_per_center,
        seed=seed,
        local_files_only=local_files_only,
    )

    train_partitions = train_dataset.get_partitions()
    test_data = test_dataset.get_datapoints()
    num_classes = full_exp5.infer_num_classes(train_partitions)
    alpha = full_exp5.build_alpha(train_partitions, num_classes)

    reduced_outputs = write_reduced_distribution_outputs(
        output_dir=run_path / "reduced_dataset",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        title_suffix=f"{experiment_name}, {condition.key}",
    )

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
    problem = benchmark.BenchmarkProblem(network=network, test_data=test_data)
    x0 = pytorch_initialization(network, all_same=True)
    metadata = {
        "center_ids": train_dataset.center_ids,
        "center_names": train_dataset.center_names,
        "class_names": train_dataset.class_names,
        "n_classes": train_dataset.n_targets,
        "n_train_samples": sum(len(partition) for partition in train_partitions),
        "n_test_samples": len(test_data),
        "sampling": {
            "sample_fraction_per_center": sample_fraction_per_center,
            "min_samples_per_center": min_samples_per_center,
            "strategy": (
                "For each official split and center, select ceil(10% of that split-center size), "
                "with a minimum of 100 samples when available, using deterministic stratified sampling by class."
            ),
            "seed": seed,
        },
        "reduced_dataset_outputs": reduced_outputs,
    }
    return problem, x0, metadata


def build_algorithm(algorithm_key: str, x0: Any, selected_hyperparameters: dict[str, Any]) -> Any:
    params = full_exp5.algorithm_hyperparameters(selected_hyperparameters, algorithm_key)
    if algorithm_key == "fedavg":
        return FedAvg(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "fedlt":
        return FedLT(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "fedopt":
        algorithm_name = full_exp5.selected_algorithm_name(selected_hyperparameters, algorithm_key)
        fedopt_class = {"FedAdam": FedAdam, "FedYogi": FedYogi, "FedAdagrad": FedAdagrad}[algorithm_name]
        return fedopt_class(iterations=iterations, selection_scheme=None, x0=x0, **params)
    if algorithm_key == "fedpd":
        params["num_local_steps"] = params.pop("num_local_epochs")
        return FedPD(iterations=iterations, x0=x0, **params)
    if algorithm_key == "scaffold":
        return Scaffold(iterations=iterations, selection_scheme=None, x0=x0, **params)
    raise ValueError(f"Unknown reduced-pilot algorithm key: {algorithm_key}")


def build_algorithms(x0: Any, selected_hyperparameters: dict[str, Any], algorithm_keys: Sequence[str]) -> list[Any]:
    return [build_algorithm(algorithm_key, x0, selected_hyperparameters) for algorithm_key in algorithm_keys]


def write_metadata(
    *,
    run_path: Path,
    condition: full_exp5.Condition,
    requested_algorithms: Sequence[str],
    selected_hyperparameters: dict[str, Any],
    dataset_metadata: dict[str, Any],
    status: dict[str, Any],
) -> None:
    metadata = {
        "experiment": experiment_name,
        "title": "Fed-ISIC2019 reduced-budget communication impairment pilot",
        "purpose": (
            "Produce time-bounded diagnostic results for clean and combined communication conditions when the "
            "full-data Experiment 5 is too expensive to finish under the deadline."
        ),
        "important_limitations": (
            "This is a reduced-budget pilot using stratified capped train/test subsets. It is not a replacement "
            "for the full Fed-ISIC2019 benchmark."
        ),
        "dataset": "Fed-ISIC2019",
        "dataset_source": "flwrlabs/fed-isic2019",
        "partition": "natural FLamby/Flower center split, then stratified capped sampling within each center",
        "evaluation_split": "stratified capped sample from the official Fed-ISIC2019 test split",
        "n_clients": len(dataset_metadata["center_ids"]),
        "n_classes": dataset_metadata["n_classes"],
        "class_names": dataset_metadata["class_names"],
        "center_ids": dataset_metadata["center_ids"],
        "center_names": dataset_metadata["center_names"],
        "n_train_samples": dataset_metadata["n_train_samples"],
        "n_test_samples": dataset_metadata["n_test_samples"],
        "sampling": dataset_metadata["sampling"],
        "reduced_dataset_outputs": dataset_metadata["reduced_dataset_outputs"],
        "n_trials": n_trials,
        "iterations": iterations,
        "state_snapshot_period": state_snapshot_period,
        "checkpoint_step": checkpoint_step,
        "batch_size": batch_size,
        "seed": seed,
        "device": device.value,
        "load_dataset": load_dataset,
        "selected_hyperparameters_path": str(full_exp5.selected_hyperparameters_path),
        "selected_hyperparameters": selected_hyperparameters,
        "model": model_name,
        "pretrained": pretrained,
        "loss": "WeightedFocalLoss(gamma=2.0, alpha=FLamby Fed-ISIC2019 weights by default)",
        "class_weight_mode": full_exp5.class_weight_mode,
        "network": {
            "participation": "full participation over active clients",
            "client_selection": None,
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
            for configured_condition in condition_lookup().values()
        ],
        "requested_algorithms": list(requested_algorithms),
        "algorithm_note": "`fedopt` resolves to the selected FedOpt-family winner, currently FedAdam.",
        "status": status,
    }
    (run_path / "experiment5_reduced_pilot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_condition(*, condition: full_exp5.Condition, run_path: Path, requested_algorithms: Sequence[str]) -> None:
    selected_hyperparameters = full_exp5.load_selected_hyperparameters()
    print(f"Running reduced pilot condition={condition.key}; results: {run_path}")
    problem, x0, dataset_metadata = build_reduced_problem(condition, run_path=run_path)
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
            "reduced_budget_pilot": True,
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
        full_exp5.save_metric_dataframes(metric_result, run_path / "metric_dataframes")
        full_exp5.save_annotated_metric_plots(metric_result, condition, run_path / "annotated_plots")
        metric_result.agent_metrics = None
        full_exp5.save_pickle_zst(metric_result, run_path / metric_result_filename)
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
        full_exp5.clear_cuda_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        default="clean_baseline",
        choices=[*condition_keys],
        help="Reduced-pilot condition key to run.",
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
        run_id = f"{run_id}_{full_exp5.slugify(args.run_label)}"
    run_path = checkpoint_root / selected_condition.key / run_id

    run_condition(condition=selected_condition, run_path=run_path, requested_algorithms=algorithm_order)
    print(f"Experiment 5 reduced pilot complete: {run_path}")


if __name__ == "__main__":
    main()
