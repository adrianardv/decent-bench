from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from decent_bench.utils.types import SupportedDevices

from experiments.fedisic2019.experiment0 import experiment0 as base


retune_algorithm_choices = ["feddyn", "fedprox", "fedlt"]


def parse_args() -> base.RuntimeConfig:
    parser = argparse.ArgumentParser(
        description="Retune selected Fed-ISIC2019 experiment0 algorithms with compact fixed candidate lists."
    )
    parser.add_argument("--algorithm", choices=retune_algorithm_choices, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--final-iterations", type=int, default=1500)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--skip-final-run", action="store_true")
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--batch-size", type=int, default=base.default_batch_size)
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
    if args.model == "small_cnn" and args.batch_size == base.default_batch_size:
        batch_size = base.default_debug_batch_size

    return base.RuntimeConfig(
        algorithm=args.algorithm,
        iterations=args.iterations,
        final_iterations=args.final_iterations,
        n_trials=args.n_trials,
        n_random_candidates=0,
        max_grid_candidates=0,
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


def retune_candidates(algorithm_key: str) -> list[base.Candidate]:
    builders = {
        "feddyn": feddyn_candidates,
        "fedprox": fedprox_candidates,
        "fedlt": fedlt_candidates,
    }
    return builders[algorithm_key]()


def feddyn_candidates() -> list[base.Candidate]:
    return [
        base.candidate(
            "feddyn",
            "FedDyn",
            "FedDyn",
            "current_best_low_lr_alpha",
            "retune",
            {"step_size": 0.0004526141826161055, "num_local_epochs": 4, "alpha": 0.011806209777155664},
        ),
        base.candidate(
            "feddyn",
            "FedDyn",
            "FedDyn",
            "lr_0p001_e4_alpha_0p01",
            "retune",
            {"step_size": 0.001, "num_local_epochs": 4, "alpha": 0.01},
        ),
        base.candidate(
            "feddyn",
            "FedDyn",
            "FedDyn",
            "lr_0p002_e4_alpha_0p01",
            "retune",
            {"step_size": 0.002, "num_local_epochs": 4, "alpha": 0.01},
        ),
        base.candidate(
            "feddyn",
            "FedDyn",
            "FedDyn",
            "lr_0p005_e3_alpha_0p05",
            "retune",
            {"step_size": 0.005, "num_local_epochs": 3, "alpha": 0.05},
        ),
        base.candidate(
            "feddyn",
            "FedDyn",
            "FedDyn",
            "lr_0p01_e2_alpha_0p1",
            "retune",
            {"step_size": 0.01, "num_local_epochs": 2, "alpha": 0.1},
        ),
        base.candidate(
            "feddyn",
            "FedDyn",
            "FedDyn",
            "lr_0p016_e3_alpha_0p33",
            "retune",
            {"step_size": 0.016013056680630116, "num_local_epochs": 3, "alpha": 0.33075447277711245},
        ),
    ]


def fedprox_candidates() -> list[base.Candidate]:
    return [
        base.candidate(
            "fedprox",
            "FedProx",
            "FedProx",
            "current_best_high_lr_mu",
            "retune",
            {"step_size": 0.03202611336126023, "num_local_epochs": 4, "mu": 0.03804421124041356},
        ),
        base.candidate(
            "fedprox",
            "FedProx",
            "FedProx",
            "top_alt_lr_0p016_e4_mu_0p0095",
            "retune",
            {"step_size": 0.016013056680630116, "num_local_epochs": 4, "mu": 0.00951105281010339},
        ),
        base.candidate(
            "fedprox",
            "FedProx",
            "FedProx",
            "flamby_lr_0p01_e4_mu_0p001",
            "retune",
            {"step_size": 0.01, "num_local_epochs": 4, "mu": 0.001},
        ),
        base.candidate(
            "fedprox",
            "FedProx",
            "FedProx",
            "flamby_lr_0p01_e2_mu_0p001",
            "retune",
            {"step_size": 0.01, "num_local_epochs": 2, "mu": 0.001},
        ),
        base.candidate(
            "fedprox",
            "FedProx",
            "FedProx",
            "lr_0p02_e3_mu_0p005",
            "retune",
            {"step_size": 0.02, "num_local_epochs": 3, "mu": 0.005},
        ),
        base.candidate(
            "fedprox",
            "FedProx",
            "FedProx",
            "lr_0p02_e4_mu_0p001",
            "retune",
            {"step_size": 0.02, "num_local_epochs": 4, "mu": 0.001},
        ),
    ]


def fedlt_candidates() -> list[base.Candidate]:
    adam_args = {"beta1": 0.5, "beta2": 0.999, "epsilon": 1e-8}
    return [
        base.candidate(
            "fedlt",
            "FedLT",
            "FedLT",
            "current_best_adam_low_rho",
            "retune",
            {
                "step_size": 0.0003897772307362001,
                "num_local_epochs": 2,
                "rho": 0.015837567625216948,
                "local_solver": "adam",
                "solver_args": adam_args,
            },
        ),
        base.candidate(
            "fedlt",
            "FedLT",
            "FedLT",
            "adam_lr_0p0015_e5_rho_1",
            "retune",
            {
                "step_size": 0.0015,
                "num_local_epochs": 5,
                "rho": 1.0,
                "local_solver": "adam",
                "solver_args": adam_args,
            },
        ),
        base.candidate(
            "fedlt",
            "FedLT",
            "FedLT",
            "adam_lr_0p003_e5_rho_1",
            "retune",
            {
                "step_size": 0.003,
                "num_local_epochs": 5,
                "rho": 1.0,
                "local_solver": "adam",
                "solver_args": adam_args,
            },
        ),
        base.candidate(
            "fedlt",
            "FedLT",
            "FedLT",
            "adam_lr_0p0015_e4_rho_0p1",
            "retune",
            {
                "step_size": 0.0015,
                "num_local_epochs": 4,
                "rho": 0.1,
                "local_solver": "adam",
                "solver_args": adam_args,
            },
        ),
        base.candidate(
            "fedlt",
            "FedLT",
            "FedLT",
            "adam_lr_0p0015_e4_rho_0p01",
            "retune",
            {
                "step_size": 0.0015,
                "num_local_epochs": 4,
                "rho": 0.01,
                "local_solver": "adam",
                "solver_args": adam_args,
            },
        ),
        base.candidate(
            "fedlt",
            "FedLT",
            "FedLT",
            "nesterov_lr_0p0015_e4_rho_0p5",
            "retune",
            {
                "step_size": 0.0015,
                "num_local_epochs": 4,
                "rho": 0.5,
                "local_solver": "nesterov",
                "solver_args": {"momentum": 0.9},
            },
        ),
    ]


def save_retune_best_hyperparameters(
    best_candidate: base.Candidate,
    best_result_row: dict[str, Any],
    path: Path,
    *,
    config: base.RuntimeConfig,
) -> None:
    payload = {
        "metadata": {
            "experiment": "experiment0_retune",
            "dataset": "Fed-ISIC2019",
            "dataset_source": "flwrlabs/fed-isic2019",
            "partition": "natural FLamby/Flower center split",
            "n_clients": 6,
            "validation_fraction_from_train": base.validation_fraction,
            "n_trials": config.n_trials,
            "iterations": config.iterations,
            "batch_size": config.batch_size,
            "seed": base.seed,
            "selection_metric": base.selection_metric,
            "tie_break_metric": base.tie_break_metric,
            "client_participation": "full",
            "communication": "no drops, no noise, no compression",
            "model": config.model_name,
            "pretrained": config.pretrained if config.model_name == "efficientnet_b0" else False,
            "class_weight_mode": config.class_weight_mode,
            "search_strategy": "compact fixed retuning candidates based on experiment0 results",
        },
        "best_hyperparameters": base.best_payload(best_candidate, best_result_row),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    config = parse_args()
    config.run_path.mkdir(parents=True, exist_ok=True)
    candidate_results_path = config.run_path / "exp0_retune_candidate_results.csv"
    best_path = config.run_path / "exp0_retune_best_hyperparameters.json"

    train_partitions, validation_data, center_ids, data_metadata = base.load_data(config)
    candidates = retune_candidates(config.algorithm)
    run_metadata = {
        **data_metadata,
        "run_config": {
            "algorithm": config.algorithm,
            "iterations": config.iterations,
            "final_iterations": config.final_iterations,
            "n_trials": config.n_trials,
            "run_final": config.run_final,
            "batch_size": config.batch_size,
            "device": config.device.value,
            "model": config.model_name,
            "pretrained": config.pretrained,
            "class_weight_mode": config.class_weight_mode,
            "client_participation": "full",
            "retune_candidate_count": len(candidates),
        },
        "retune_candidates": [base.best_payload(candidate, row=None) for candidate in candidates],
    }
    (config.run_path / "exp0_retune_dataset_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Experiment 0 retune run path: {config.run_path}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Running compact retuning search with {len(candidates)} candidates.")

    rows = base.run_candidate_list(
        candidates,
        train_partitions,
        validation_data,
        center_ids,
        config=config,
        candidate_results_path=candidate_results_path,
        starting_index=1,
    )

    final_best_row = base.best_row(rows)
    final_best_candidate = base.row_to_candidate(final_best_row, candidates)
    save_retune_best_hyperparameters(final_best_candidate, final_best_row, best_path, config=config)

    if config.run_final:
        print(f"Running final best-candidate curve for {config.final_iterations} iterations.")
        base.run_final_curve(final_best_candidate, train_partitions, validation_data, center_ids, config=config)

    print(f"Candidate results saved to: {candidate_results_path}")
    print(f"Best hyperparameters saved to: {best_path}")


if __name__ == "__main__":
    main()
