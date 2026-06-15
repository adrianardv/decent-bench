# ruff: noqa: ANN401, E402, INP001, T201
"""
Quick FedLT lr=0.001 candidate curve diagnostic for Fed-ISIC2019.

This script runs the same intermediate FedLT candidate as the middle-candidate
diagnostic, but with step_size=0.001.
"""

from __future__ import annotations

import argparse
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


default_iterations = 1500
default_n_trials = 1
default_state_snapshot_period = 150
checkpoint_root = Path("experiments/fedisic2019/checkpoints/experiment0/fedlt")
metric_result_filename = "metric_computation.pkl.zst"


def diagnostic_candidate() -> base.Candidate:
    return base.candidate(
        "fedlt",
        "FedLT",
        "FedLT",
        "lr_0p001_e3_rho_0p05",
        "diagnostic",
        {
            "step_size": 0.001,
            "num_local_epochs": 3,
            "rho": 0.05,
            "local_solver": "adam",
            "solver_args": {"beta1": 0.5, "beta2": 0.999, "epsilon": 1e-8},
        },
    )


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
        algorithm="fedlt",
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


def save_metric_dataframes(metric_result: Any, output_path: Path) -> None:
    table_frame, plot_frame = metric_result.to_dataframe()
    dataframes_path = output_path / "metric_dataframes"
    dataframes_path.mkdir(parents=True, exist_ok=True)
    if table_frame is not None:
        table_frame.to_csv(dataframes_path / "table_metrics.csv", index=False)
    if plot_frame is not None:
        plot_frame.to_csv(dataframes_path / "plot_metrics.csv", index=False)


def save_candidate_summary(metric_result: Any, candidate: base.Candidate, output_path: Path) -> None:
    table_frame, _ = metric_result.to_dataframe()
    if table_frame is None:
        return
    summary: dict[str, Any] = {
        "candidate": base.best_payload(candidate, row=None),
        "metrics": {},
    }
    for metric_name, statistic in (
        ("server balanced accuracy", None),
        ("server accuracy", None),
        ("loss", "avg"),
    ):
        mean, margin = base.metric_value(table_frame, metric_name, statistic=statistic)
        summary["metrics"][metric_name if statistic is None else f"{metric_name}_{statistic}"] = {
            "mean": mean,
            "margin_of_error": margin,
        }
    (output_path / "fedlt_lr_0p001_candidate_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run_name = args.run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}_lr_0p001_curve"
    run_path = checkpoint_root / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    config = runtime_config(args, run_path)
    candidate = diagnostic_candidate()

    train_partitions, validation_data, center_ids, data_metadata = base.load_data(config)
    problem, x0 = base.build_problem(
        train_partitions,
        validation_data,
        center_ids,
        config=config,
        state_snapshot_period=args.state_snapshot_period,
    )
    algorithm = base.build_algorithm(candidate, x0, args.iterations)
    algorithm.name = "FedLT lr 0.001 candidate"

    metadata = {
        **data_metadata,
        "experiment": "fedlt_lr_0p001_candidate_curve",
        "purpose": "Quickly test the FedLT intermediate candidate with step_size=0.001.",
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
        "candidate": base.best_payload(candidate, row=None),
    }
    (run_path / "fedlt_lr_0p001_candidate_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"FedLT lr=0.001 candidate run path: {run_path}")
    print(f"Running 1 FedLT candidate for {args.iterations} iterations and {args.n_trials} trial(s).")
    result = benchmark.benchmark(
        algorithms=[algorithm],
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
    save_metric_dataframes(metric_result, run_path)
    save_candidate_summary(metric_result, candidate, run_path)
    metric_result.agent_metrics = None
    base.save_pickle_zst(metric_result, run_path / metric_result_filename)
    (run_path / "metric_computation_complete.json").write_text(
        json.dumps({"metric_computation_complete": True}, indent=2),
        encoding="utf-8",
    )
    base.cleanup_cuda()
    print(f"FedLT lr=0.001 candidate diagnostic complete: {run_path}")


if __name__ == "__main__":
    main()
