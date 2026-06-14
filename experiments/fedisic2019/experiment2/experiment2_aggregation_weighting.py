# ruff: noqa: ANN401, D103, E402, INP001, PLR0911, T201
"""
Fed-ISIC2019 aggregation-weighting benchmark.

For each tuned federated algorithm, this experiment compares uniform client
aggregation against data-size weighted aggregation under a clean cross-silo
baseline.
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
from decent_bench.utils.checkpoint_manager import CheckpointManager
from experiments.fedisic2019.experiment5 import experiment5_communication_impairments as base


experiment_group = "experiment2"
experiment_name = "experiment2_aggregation_weighting"
checkpoint_root = Path("experiments/fedisic2019/checkpoints") / experiment_group
aggregation_variants = (False, True)


def build_algorithm(
    algorithm_key: str,
    x0: Any,
    selected_hyperparameters: dict[str, Any],
    *,
    weighted_aggregation: bool,
) -> Any:
    params = base.algorithm_hyperparameters(selected_hyperparameters, algorithm_key)
    aggregation_label = "data-size weighted" if weighted_aggregation else "uniform"

    if algorithm_key == "fedavg":
        return FedAvg(
            iterations=base.iterations,
            selection_scheme=None,
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedAvg {aggregation_label}",
            **params,
        )
    if algorithm_key == "fedprox":
        return FedProx(
            iterations=base.iterations,
            selection_scheme=None,
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedProx {aggregation_label}",
            **params,
        )
    if algorithm_key == "scaffold":
        return Scaffold(
            iterations=base.iterations,
            selection_scheme=None,
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"SCAFFOLD {aggregation_label}",
            **params,
        )
    if algorithm_key == "fednova":
        params["num_local_steps"] = params.pop("num_local_epochs")
        return FedNova(
            iterations=base.iterations,
            selection_scheme=None,
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedNova {aggregation_label}",
            **params,
        )
    if algorithm_key == "fedopt":
        algorithm_name = base.selected_algorithm_name(selected_hyperparameters, algorithm_key)
        fedopt_class = {"FedAdam": FedAdam, "FedYogi": FedYogi, "FedAdagrad": FedAdagrad}[algorithm_name]
        return fedopt_class(
            iterations=base.iterations,
            selection_scheme=None,
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"{algorithm_name} {aggregation_label}",
            **params,
        )
    if algorithm_key == "fedlt":
        return FedLT(
            iterations=base.iterations,
            selection_scheme=None,
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedLT {aggregation_label}",
            **params,
        )
    if algorithm_key == "feddyn":
        return FedDyn(
            iterations=base.iterations,
            selection_scheme=None,
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedDyn {aggregation_label}",
            **params,
        )
    if algorithm_key == "fedpd":
        params["num_local_steps"] = params.pop("num_local_epochs")
        return FedPD(
            iterations=base.iterations,
            x0=x0,
            weighted_aggregation=weighted_aggregation,
            name=f"FedPD {aggregation_label}",
            **params,
        )
    raise ValueError(f"Unknown algorithm key: {algorithm_key}")


def build_algorithms(algorithm_key: str, x0: Any, selected_hyperparameters: dict[str, Any]) -> list[Any]:
    return [
        build_algorithm(
            algorithm_key,
            x0,
            selected_hyperparameters,
            weighted_aggregation=weighted_aggregation,
        )
        for weighted_aggregation in aggregation_variants
    ]


def write_metadata(
    *,
    run_path: Path,
    algorithm_key: str,
    selected_hyperparameters: dict[str, Any],
    dataset_metadata: dict[str, Any],
    status: dict[str, Any],
) -> None:
    train_counts = [
        {"center": 0, "center_name": "BCN", "train": 9930},
        {"center": 1, "center_name": "HAM_vidir_molemax", "train": 3163},
        {"center": 2, "center_name": "HAM_vidir_modern", "train": 2691},
        {"center": 3, "center_name": "HAM_rosendahl", "train": 1807},
        {"center": 4, "center_name": "MSK", "train": 655},
        {"center": 5, "center_name": "HAM_vienna_dias", "train": 351},
    ]
    metadata = {
        "experiment": experiment_name,
        "title": "Fed-ISIC2019 aggregation weighting benchmark",
        "purpose": (
            "Compare uniform and data-size weighted aggregation for each tuned federated algorithm. "
            "Fed-ISIC2019 has strong quantity skew across its six cross-silo clients, so data-size weighting may "
            "have a larger performance impact than in FEMNIST."
        ),
        "execution": "one algorithm pair per launch; each pair writes to experiment2/<algorithm>/run_<timestamp>",
        "dataset": "Fed-ISIC2019",
        "dataset_source": "flwrlabs/fed-isic2019",
        "partition": "natural FLamby/Flower center split",
        "evaluation_split": "official Fed-ISIC2019 test split",
        "n_clients": len(dataset_metadata["center_ids"]),
        "n_classes": dataset_metadata["n_classes"],
        "class_names": dataset_metadata["class_names"],
        "center_ids": dataset_metadata["center_ids"],
        "center_names": dataset_metadata["center_names"],
        "n_train_samples": dataset_metadata["n_train_samples"],
        "n_test_samples": dataset_metadata["n_test_samples"],
        "quantity_skew": {
            "train_counts_per_center": train_counts,
            "largest_train_center": "BCN",
            "smallest_train_center": "HAM_vienna_dias",
            "largest_to_smallest_train_ratio": 9930 / 351,
            "summary": (
                "Official train split center sizes range from 351 to 9930 samples, "
                "a roughly 28.3x largest/smallest ratio."
            ),
        },
        "n_trials": base.n_trials,
        "iterations": base.iterations,
        "state_snapshot_period": base.state_snapshot_period,
        "checkpoint_step": base.checkpoint_step,
        "batch_size": base.batch_size,
        "seed": base.seed,
        "device": base.device.value,
        "load_dataset": base.load_dataset,
        "selected_hyperparameters_path": str(base.selected_hyperparameters_path),
        "selected_hyperparameters": selected_hyperparameters,
        "model": base.model_name,
        "pretrained": base.pretrained,
        "loss": "WeightedFocalLoss(gamma=2.0, alpha=FLamby Fed-ISIC2019 weights by default)",
        "class_weight_mode": base.class_weight_mode,
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
        "requested_algorithm": algorithm_key,
        "status": status,
    }
    (run_path / "experiment2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_algorithm_pair(*, algorithm_key: str, run_path: Path) -> None:
    selected_hyperparameters = base.load_selected_hyperparameters()
    clean_condition = base.condition_lookup()["clean_baseline"]
    print(f"Running {algorithm_key}; results: {run_path}")
    problem, x0, dataset_metadata = base.build_problem(clean_condition)
    algorithms = build_algorithms(algorithm_key, x0, selected_hyperparameters)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=run_path,
        checkpoint_step=base.checkpoint_step,
        keep_n_checkpoints=1,
        benchmark_metadata={
            "experiment": experiment_name,
            "algorithm": algorithm_key,
            "aggregation_variants": [algorithm.name for algorithm in algorithms],
            "n_trials": base.n_trials,
            "iterations": base.iterations,
            "state_snapshot_period": base.state_snapshot_period,
            "checkpoint_step": base.checkpoint_step,
        },
    )

    status: dict[str, Any] = {
        "algorithm": algorithm_key,
        "status": "started",
        "variants": [algorithm.name for algorithm in algorithms],
    }
    try:
        result = benchmark.benchmark(
            algorithms=algorithms,
            benchmark_problem=problem,
            n_trials=base.n_trials,
            max_processes=1,
            progress_step=base.progress_step,
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
            show_plots=base.show_plots,
            log_level=logging.INFO,
        )
        base.save_metric_dataframes(metric_result, checkpoint_manager.get_results_path())
        metric_result.agent_metrics = None
        base.save_pickle_zst(metric_result, run_path / base.metric_result_filename)
        (run_path / "metric_computation_complete.json").write_text(
            json.dumps({"metric_computation_complete": True}, indent=2),
            encoding="utf-8",
        )
        status = {
            "algorithm": algorithm_key,
            "status": "ok",
            "variants": [algorithm.name for algorithm in algorithms],
        }
    except Exception as error:
        status = {
            "algorithm": algorithm_key,
            "status": "failed",
            "variants": [algorithm.name for algorithm in algorithms],
            "error": repr(error),
        }
        raise
    finally:
        write_metadata(
            run_path=run_path,
            algorithm_key=algorithm_key,
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
        base.clear_cuda_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        default="fedavg",
        choices=[*base.algorithm_order],
        help="Federated algorithm to run.",
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
    run_id = f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    if args.run_label:
        run_id = f"{run_id}_{base.slugify(args.run_label)}"
    run_path = checkpoint_root / args.algorithm / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    run_algorithm_pair(algorithm_key=args.algorithm, run_path=run_path)
    print(f"Experiment 2 aggregation-weighting benchmark complete: {run_path}")


if __name__ == "__main__":
    main()
