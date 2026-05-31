"""Collect drift-aware FedLT version runs and select the best candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd

import experiment0 as exp0


SUMMARY_FIELDS = [
    "status",
    "version_index",
    "candidate_id",
    "variant",
    "score",
    "server_accuracy_mean",
    "server_accuracy_margin_of_error",
    "accuracy_avg_mean",
    "accuracy_avg_margin_of_error",
    "validation_loss_mean",
    "validation_loss_margin_of_error",
    "client_drift_avg_mean",
    "client_drift_avg_margin_of_error",
    "client_drift_max_mean",
    "client_drift_max_margin_of_error",
    "elapsed_seconds",
    "step_size",
    "num_local_epochs",
    "rho",
    "local_solver",
    "solver_args.beta1",
    "solver_args.beta2",
    "solver_args.epsilon",
    "error",
]


def finite_float(value: Any) -> float | None:
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    return float_value if math.isfinite(float_value) else None


def is_selectable(row: dict[str, Any]) -> bool:
    required_fields = [
        "server_accuracy_mean",
        "server_accuracy_margin_of_error",
        "validation_loss_mean",
        "client_drift_avg_mean",
    ]
    return row.get("status") == "ok" and all(finite_float(row.get(field)) is not None for field in required_fields)


def score(row: dict[str, Any]) -> float:
    """Accuracy-first score with a small drift penalty.

    The accuracy lower confidence bound remains dominant. Drift is used to
    distinguish candidates that are close in server accuracy.
    """

    accuracy_lcb = float(row["server_accuracy_mean"]) - float(row["server_accuracy_margin_of_error"])
    drift = float(row["client_drift_avg_mean"])
    return accuracy_lcb - 0.01 * drift


def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["score"]),
        -float(row["client_drift_avg_mean"]),
        -float(row["validation_loss_mean"]),
        float(row.get("accuracy_avg_mean") or 0.0),
    )


def load_rows(sweep_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(sweep_path.glob("version_*/version_result.json")):
        row = json.loads(result_path.read_text(encoding="utf-8"))
        row["version_index"] = result_path.parent.name.removeprefix("version_")
        if is_selectable(row):
            row["score"] = score(row)
        else:
            row["score"] = ""
        rows.append(row)
    return rows


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_best_artifacts(best_version_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    for relative_path in [
        Path("metadata.json"),
        Path("version_result.json"),
        Path(exp0.metric_result_filename),
        Path("results") / "plot.png",
        Path("results") / "table.tex",
        Path("results") / "table.txt",
    ]:
        source = best_version_path / relative_path
        if source.exists():
            target = output_path / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def write_best_hyperparameters(best_row: dict[str, Any], output_path: Path) -> None:
    payload = {
        "metadata": {
            "experiment": "experiment0",
            "algorithm": "fedlt",
            "dataset": "FEMNIST",
            "search_strategy": "parallel drift-aware FedLT grid",
            "selection_rule": (
                "Reject failed/non-finite versions. Rank by server accuracy lower confidence bound "
                "minus 0.01 * average client drift, then lower average drift, lower validation loss, "
                "and higher average client accuracy."
            ),
        },
        "best_hyperparameters": {
            "algorithm_name": "FedLT",
            "variant": best_row["variant"],
            "search_stage": best_row["search_stage"],
            "hyperparameters": {
                "step_size": float(best_row["step_size"]),
                "num_local_epochs": int(best_row["num_local_epochs"]),
                "rho": float(best_row["rho"]),
                "local_solver": best_row["local_solver"],
                "solver_args": {
                    "beta1": float(best_row["solver_args.beta1"]),
                    "beta2": float(best_row["solver_args.beta2"]),
                    "epsilon": float(best_row["solver_args.epsilon"]),
                },
            },
            "server_accuracy_mean": best_row["server_accuracy_mean"],
            "server_accuracy_margin_of_error": best_row["server_accuracy_margin_of_error"],
            "accuracy_avg_mean": best_row["accuracy_avg_mean"],
            "accuracy_avg_margin_of_error": best_row["accuracy_avg_margin_of_error"],
            "validation_loss_mean": best_row["validation_loss_mean"],
            "validation_loss_margin_of_error": best_row["validation_loss_margin_of_error"],
            "client_drift_avg_mean": best_row["client_drift_avg_mean"],
            "client_drift_avg_margin_of_error": best_row["client_drift_avg_margin_of_error"],
            "client_drift_max_mean": best_row["client_drift_max_mean"],
            "client_drift_max_margin_of_error": best_row["client_drift_max_margin_of_error"],
            "selected_solver": best_row["local_solver"],
        },
    }
    (output_path / "exp0_best_hyperparameters.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_path", type=Path, help="Path containing version_*/version_result.json files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.sweep_path)
    if not rows:
        raise RuntimeError(f"No version results found under {args.sweep_path}")

    selectable_rows = [row for row in rows if is_selectable(row)]
    if not selectable_rows:
        raise RuntimeError("No selectable FedLT versions found.")

    ranked_rows = sorted(selectable_rows, key=sort_key, reverse=True)
    all_rows_by_rank = ranked_rows + [row for row in rows if not is_selectable(row)]

    summary_path = args.sweep_path / "fedlt_drift_summary.csv"
    write_summary(all_rows_by_rank, summary_path)
    pd.DataFrame.from_records(all_rows_by_rank).to_json(
        args.sweep_path / "fedlt_drift_summary.json",
        orient="records",
        indent=2,
    )

    best_row = ranked_rows[0]
    best_version_path = args.sweep_path / f"version_{int(best_row['version_index']):03d}"
    best_output_path = args.sweep_path / "best_version"
    copy_best_artifacts(best_version_path, best_output_path)
    write_best_hyperparameters(best_row, best_output_path)

    print(f"Summary: {summary_path}")
    print(f"Best version: {best_version_path}")
    print(f"Best artifacts: {best_output_path}")
    print(
        "Best metrics: "
        f"server_accuracy={float(best_row['server_accuracy_mean']):.4f}, "
        f"accuracy_avg={float(best_row['accuracy_avg_mean']):.4f}, "
        f"loss={float(best_row['validation_loss_mean']):.4f}, "
        f"client_drift_avg={float(best_row['client_drift_avg_mean']):.4f}"
    )


if __name__ == "__main__":
    main()
