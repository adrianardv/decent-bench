# ruff: noqa: ANN401, E402, INP001, T201
"""
Quick FedDyn curve diagnostic for Fed-ISIC2019 experiment0 retuning.

This script benchmarks the FedDyn retune candidates that had reasonable final
balanced accuracy but may have different curve shapes. It is intentionally
small: 1 trial, 500 iterations by default, and only three candidate variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from decent_bench import benchmark
from decent_bench.metrics import metric_library as ml
from decent_bench.utils.types import SupportedDevices

from experiments.fedisic2019.experiment0 import experiment0 as base
from experiments.fedisic2019.experiment0 import experiment0_retune


default_iterations = 500
default_n_trials = 1
default_state_snapshot_period = 50
checkpoint_root = Path("experiments/fedisic2019/checkpoints/experiment0/feddyn")
metric_result_filename = "metric_computation.pkl.zst"


def selected_candidates() -> list[base.Candidate]:
    wanted_variants = {
        "lr_0p001_e4_alpha_0p01",
        "lr_0p002_e4_alpha_0p01",
        "lr_0p016_e3_alpha_0p33",
    }
    return [
        candidate
        for candidate in experiment0_retune.feddyn_candidates()
        if candidate.variant in wanted_variants
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=default_iterations)
    parser.add_argument("--n-trials", type=int, default=default_n_trials)
    parser.add_argument("--state-snapshot-period", type=int, default=default_state_snapshot_period)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--batch-size", type=int, default=base.default_batch_size)
    parser.add_argument("--device", choices=[device.value for device in SupportedDevices], default=SupportedDevices.GPU.value)
    parser.add_argument("--model", choices=["efficientnet_b0", "small_cnn"], default="efficientnet_b0")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--class-weight-mode", choices=["flamby", "computed"], default="flamby")
    parser.add_argument("--max-samples-per-client", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--load-dataset", action="store_true", help="Materialize lazy image datasets inside PyTorchCost.")
    return parser.parse_args()


def runtime_config(args: argparse.Namespace, run_path: Path) -> base.RuntimeConfig:
    batch_size = args.batch_size
    if args.model == "small_cnn" and args.batch_size == base.default_batch_size:
        batch_size = base.default_debug_batch_size
    return base.RuntimeConfig(
        algorithm="feddyn",
        iterations=args.iterations,
        final_iterations=args.iterations,
        n_trials=args.n_trials,
        n_random_candidates=0,
        max_grid_candidates=0,
        run_final=False,
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


def build_algorithms(candidates: list[base.Candidate], x0: Any, iterations: int) -> list[Any]:
    algorithms = []
    for candidate in candidates:
        algorithm = base.build_algorithm(candidate, x0, iterations)
        algorithm.name = f"FedDyn {candidate.variant}"
        algorithms.append(algorithm)
    return algorithms


def save_table_frame(metric_result: Any, output_path: Path) -> None:
    table_frame, plot_frame = metric_result.to_dataframe()
    dataframes_path = output_path / "metric_dataframes"
    dataframes_path.mkdir(parents=True, exist_ok=True)
    if table_frame is not None:
        table_frame.to_csv(dataframes_path / "table_metrics.csv", index=False)
    if plot_frame is not None:
        plot_frame.to_csv(dataframes_path / "plot_metrics.csv", index=False)


def save_candidate_summary(metric_result: Any, candidates: list[base.Candidate], output_path: Path) -> None:
    table_frame, _ = metric_result.to_dataframe()
    if table_frame is None:
        return
    rows = []
    candidate_by_name = {f"FedDyn {candidate.variant}": candidate for candidate in candidates}
    for algorithm_name, candidate in candidate_by_name.items():
        algorithm_rows = table_frame[table_frame["algorithm"] == algorithm_name]
        if algorithm_rows.empty:
            continue
        server_balanced_accuracy = base.metric_value(algorithm_rows, "server balanced accuracy")
        server_accuracy = base.metric_value(algorithm_rows, "server accuracy")
        validation_loss = base.metric_value(algorithm_rows, "loss", statistic="avg")
        rows.append(
            {
                "algorithm_name": algorithm_name,
                "variant": candidate.variant,
                "candidate_id": candidate.candidate_id,
                "server_balanced_accuracy_mean": server_balanced_accuracy[0],
                "server_balanced_accuracy_margin_of_error": server_balanced_accuracy[1],
                "server_accuracy_mean": server_accuracy[0],
                "validation_loss_mean": validation_loss[0],
                "validation_loss_margin_of_error": validation_loss[1],
                **base.flatten_hyperparameters(candidate.hyperparameters),
            }
        )
    if not rows:
        return
    fieldnames = list(rows[0])
    with (output_path / "feddyn_curve_diagnostic_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run_name = args.run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}_curve_diagnostic"
    run_path = checkpoint_root / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    config = runtime_config(args, run_path)

    candidates = selected_candidates()
    train_partitions, validation_data, center_ids, data_metadata = base.load_data(config)
    problem, x0 = base.build_problem(
        train_partitions,
        validation_data,
        center_ids,
        config=config,
        state_snapshot_period=args.state_snapshot_period,
    )
    algorithms = build_algorithms(candidates, x0, args.iterations)

    metadata = {
        **data_metadata,
        "experiment": "feddyn_curve_diagnostic",
        "purpose": "Compare curve shapes for FedDyn retune candidates with reasonable final balanced accuracy.",
        "iterations": args.iterations,
        "n_trials": args.n_trials,
        "state_snapshot_period": args.state_snapshot_period,
        "batch_size": config.batch_size,
        "device": config.device.value,
        "model": config.model_name,
        "pretrained": config.pretrained,
        "class_weight_mode": config.class_weight_mode,
        "client_participation": "full",
        "communication": "no drops, no noise, no compression",
        "candidates": [base.best_payload(candidate, row=None) for candidate in candidates],
    }
    (run_path / "feddyn_curve_diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"FedDyn curve diagnostic run path: {run_path}")
    print(f"Running {len(algorithms)} FedDyn candidates for {args.iterations} iterations and {args.n_trials} trial(s).")
    result = benchmark.benchmark(
        algorithms=algorithms,
        benchmark_problem=problem,
        n_trials=args.n_trials,
        max_processes=1,
        progress_step=max(1, args.iterations // 10),
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
        plot_metrics=[
            ml.ServerBalancedAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.ServerAccuracy(fmt=".2%", x_log=False, y_log=False),
            ml.Loss([np.average]),
        ],
        log_level=logging.INFO,
    )
    benchmark.display_metrics(metrics_result=metric_result, save_path=run_path / "results", show_plots=False)
    save_table_frame(metric_result, run_path)
    save_candidate_summary(metric_result, candidates, run_path)
    metric_result.agent_metrics = None
    base.save_pickle_zst(metric_result, run_path / metric_result_filename)
    (run_path / "metric_computation_complete.json").write_text(
        json.dumps({"metric_computation_complete": True}, indent=2),
        encoding="utf-8",
    )
    base.cleanup_cuda()
    print(f"FedDyn curve diagnostic complete: {run_path}")


if __name__ == "__main__":
    main()
