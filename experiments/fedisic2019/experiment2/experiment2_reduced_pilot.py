# ruff: noqa: ANN401, D103, E402, INP001, T201
"""
Fed-ISIC2019 reduced-budget aggregation-weighting pilot.

This pilot compares FedAvg uniform aggregation against data-size weighted
aggregation on a deterministic stratified proportional Fed-ISIC2019 subset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from decent_bench import benchmark
from decent_bench.algorithms.federated import FedAvg
from decent_bench.utils.checkpoint_manager import CheckpointManager
from experiments.fedisic2019.experiment5 import experiment5_communication_impairments as full_exp5
from experiments.fedisic2019.experiment5 import experiment5_reduced_pilot as reduced_exp5


experiment_group = "experiment2_reduced_pilot"
experiment_name = "experiment2_reduced_pilot"
checkpoint_root = Path("experiments/fedisic2019/checkpoints") / experiment_group
aggregation_variants = (False, True)


def build_fedavg_variants(x0: Any, selected_hyperparameters: dict[str, Any]) -> list[FedAvg]:
    params = full_exp5.algorithm_hyperparameters(selected_hyperparameters, "fedavg")
    return [
        FedAvg(
            iterations=reduced_exp5.iterations,
            selection_scheme=None,
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name="FedAvg data-size weighted" if weighted_aggregation else "FedAvg uniform",
            **params,
        )
        for weighted_aggregation in aggregation_variants
    ]


def write_metadata(
    *,
    run_path: Path,
    selected_hyperparameters: dict[str, Any],
    dataset_metadata: dict[str, Any],
    status: dict[str, Any],
) -> None:
    metadata = {
        "experiment": experiment_name,
        "title": "Fed-ISIC2019 reduced-budget aggregation weighting pilot",
        "purpose": (
            "Quickly compare FedAvg uniform aggregation against data-size weighted aggregation on a stratified "
            "capped Fed-ISIC2019 subset."
        ),
        "important_limitations": (
            "This is a reduced-budget pilot using stratified capped train/test subsets. It is not a replacement "
            "for the full-data Experiment 2."
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
        "n_trials": reduced_exp5.n_trials,
        "iterations": reduced_exp5.iterations,
        "state_snapshot_period": reduced_exp5.state_snapshot_period,
        "checkpoint_step": reduced_exp5.checkpoint_step,
        "batch_size": reduced_exp5.batch_size,
        "seed": reduced_exp5.seed,
        "device": reduced_exp5.device.value,
        "load_dataset": reduced_exp5.load_dataset,
        "selected_hyperparameters_path": str(full_exp5.selected_hyperparameters_path),
        "selected_hyperparameters": selected_hyperparameters,
        "model": reduced_exp5.model_name,
        "pretrained": reduced_exp5.pretrained,
        "loss": "WeightedFocalLoss(gamma=2.0, alpha=FLamby Fed-ISIC2019 weights by default)",
        "class_weight_mode": full_exp5.class_weight_mode,
        "network": {
            "participation": "full participation",
            "client_selection": None,
            "activation": "AlwaysActive",
            "drops": "NoDrops",
            "noise": "NoNoise",
            "compression": "NoCompression",
        },
        "aggregation_variants": {
            "uniform": "Each received client upload has equal aggregation weight.",
            "data-size weighted": "Each received client upload is weighted by its local training sample count.",
        },
        "requested_algorithm": "fedavg",
        "status": status,
    }
    (run_path / "experiment2_reduced_pilot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_fedavg_pair(*, run_path: Path) -> None:
    selected_hyperparameters = full_exp5.load_selected_hyperparameters()
    clean_condition = full_exp5.condition_lookup()["clean_baseline"]
    print(f"Running reduced pilot FedAvg aggregation pair; results: {run_path}")
    problem, x0, dataset_metadata = reduced_exp5.build_reduced_problem(clean_condition, run_path=run_path)
    algorithms = build_fedavg_variants(x0, selected_hyperparameters)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=run_path,
        checkpoint_step=reduced_exp5.checkpoint_step,
        keep_n_checkpoints=1,
        benchmark_metadata={
            "experiment": experiment_name,
            "algorithm": "fedavg",
            "aggregation_variants": [algorithm.name for algorithm in algorithms],
            "n_trials": reduced_exp5.n_trials,
            "iterations": reduced_exp5.iterations,
            "state_snapshot_period": reduced_exp5.state_snapshot_period,
            "checkpoint_step": reduced_exp5.checkpoint_step,
            "reduced_budget_pilot": True,
        },
    )

    status: dict[str, Any] = {
        "algorithm": "fedavg",
        "status": "started",
        "variants": [algorithm.name for algorithm in algorithms],
    }
    try:
        result = benchmark.benchmark(
            algorithms=algorithms,
            benchmark_problem=problem,
            n_trials=reduced_exp5.n_trials,
            max_processes=1,
            progress_step=reduced_exp5.progress_step,
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
            show_plots=reduced_exp5.show_plots,
            log_level=logging.INFO,
        )
        full_exp5.save_metric_dataframes(metric_result, run_path / "metric_dataframes")
        metric_result.agent_metrics = None
        full_exp5.save_pickle_zst(metric_result, run_path / reduced_exp5.metric_result_filename)
        (run_path / "metric_computation_complete.json").write_text(
            json.dumps({"metric_computation_complete": True}, indent=2),
            encoding="utf-8",
        )
        status = {
            "algorithm": "fedavg",
            "status": "ok",
            "variants": [algorithm.name for algorithm in algorithms],
        }
    except Exception as error:
        status = {
            "algorithm": "fedavg",
            "status": "failed",
            "variants": [algorithm.name for algorithm in algorithms],
            "error": repr(error),
        }
        raise
    finally:
        write_metadata(
            run_path=run_path,
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
        "--run-label",
        default=None,
        help="Optional suffix for the output run directory.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run_id = f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    if args.run_label:
        run_id = f"{run_id}_{full_exp5.slugify(args.run_label)}"
    run_path = checkpoint_root / "fedavg" / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    run_fedavg_pair(run_path=run_path)
    print(f"Experiment 2 reduced pilot complete: {run_path}")


if __name__ == "__main__":
    main()
