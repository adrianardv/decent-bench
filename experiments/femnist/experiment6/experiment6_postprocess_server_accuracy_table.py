# ruff: noqa: ANN401, D103, INP001, T201
"""
Build the Experiment 6 server-accuracy table across algorithms.

By default this scans the latest run under each algorithm directory and writes a
table with algorithms as rows and aggregation variants as columns.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import zstandard as zstd

checkpoint_root = Path("experiments/femnist/checkpoints/experiment6")
default_output_root = Path("experiments/femnist/results/experiment6/server_accuracy_tables")
metric_result_filename = "metric_computation.pkl.zst"

algorithm_order = (
    "fedavg",
    "fedprox",
    "scaffold",
    "fednova",
    "fedopt",
    "fedlt",
    "feddyn",
)

algorithm_labels = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "SCAFFOLD",
    "fednova": "FedNova",
    "fedopt": "FedAdam",
    "fedlt": "FedLT",
    "feddyn": "FedDyn",
}


def latest_run_dir(root: Path, algorithm_key: str) -> Path | None:
    candidates = sorted((root / algorithm_key).glob("run_*"))
    return candidates[-1] if candidates else None


def run_dir_for_algorithm(root: Path, algorithm_key: str, run_id: str | None) -> Path | None:
    if run_id is not None:
        candidate = root / algorithm_key / run_id
        return candidate if candidate.exists() else None
    return latest_run_dir(root, algorithm_key)


def load_metric_result(path: Path) -> Any:
    decompressed = zstd.ZstdDecompressor().decompress(path.read_bytes())
    return pickle.loads(decompressed)  # noqa: S301


def aggregation_label(algorithm_name: str) -> str:
    if "data-size weighted" in algorithm_name:
        return "data-size weighted"
    if "uniform" in algorithm_name:
        return "uniform"
    raise ValueError(f"Cannot infer aggregation label from algorithm name: {algorithm_name}")


def server_accuracy_from_metric_file(path: Path, algorithm_key: str) -> dict[str, Any]:
    metric_result = load_metric_result(path)
    table_df, _ = metric_result.to_dataframe()
    if table_df is None:
        raise ValueError(f"No table metrics found in {path}")

    server_accuracy = table_df[(table_df["metric"] == "server accuracy") & (table_df["statistic"].isna())]
    if server_accuracy.empty:
        server_accuracy = table_df[(table_df["metric"] == "server accuracy") & (table_df["statistic"] == "")]

    row: dict[str, Any] = {
        "algorithm": algorithm_labels[algorithm_key],
        "algorithm_key": algorithm_key,
        "uniform": float("nan"),
        "data-size weighted": float("nan"),
        "uniform margin_of_error": float("nan"),
        "data-size weighted margin_of_error": float("nan"),
        "metric_file": str(path),
    }
    for _, metric_row in server_accuracy.iterrows():
        column = aggregation_label(str(metric_row["algorithm"]))
        row[column] = float(metric_row["mean"])
        row[f"{column} margin_of_error"] = float(metric_row["margin_of_error"])
    return row


def collect_server_accuracy(
    root: Path,
    algorithms: list[str],
    run_id: str | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    missing: dict[str, str] = {}

    for algorithm_key in algorithms:
        run_dir = run_dir_for_algorithm(root, algorithm_key, run_id)
        if run_dir is None:
            missing[algorithm_key] = f"No run directory found under {root / algorithm_key}"
            continue

        metric_file = run_dir / metric_result_filename
        if not metric_file.exists():
            missing[algorithm_key] = f"No {metric_result_filename} found in {run_dir}"
            continue

        rows.append(server_accuracy_from_metric_file(metric_file, algorithm_key))

    if not rows:
        raise RuntimeError(f"No Experiment 6 metric files found under {root}")

    table = pd.DataFrame(rows)
    order_lookup = {algorithm: index for index, algorithm in enumerate(algorithm_order)}
    table["_sort_order"] = table["algorithm_key"].map(order_lookup)
    table = table.sort_values("_sort_order").drop(columns=["_sort_order"])
    return table, missing


def write_outputs(table: pd.DataFrame, output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "experiment6_server_accuracy_by_aggregation.csv", index=False)

    formatted = table[
        ["algorithm", "uniform", "data-size weighted", "uniform margin_of_error", "data-size weighted margin_of_error"]
    ].copy()
    for column in ("uniform", "data-size weighted", "uniform margin_of_error", "data-size weighted margin_of_error"):
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    formatted.to_markdown(str(output_dir / "experiment6_server_accuracy_by_aggregation.md"), index=False)
    formatted.to_latex(str(output_dir / "experiment6_server_accuracy_by_aggregation.tex"), index=False)

    (output_dir / "experiment6_server_accuracy_table_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=checkpoint_root,
        help="Root containing Experiment 6 algorithm run folders.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(algorithm_order),
        choices=list(algorithm_order),
        help="Algorithms to include.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional exact run directory name to use under every algorithm, e.g. run_20260609_120000.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a timestamped folder under experiments/femnist/results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_root / f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    table, missing = collect_server_accuracy(args.checkpoint_root, args.algorithms, args.run_id)
    metadata = {
        "checkpoint_root": str(args.checkpoint_root),
        "requested_algorithms": args.algorithms,
        "run_id": args.run_id,
        "missing_algorithms": missing,
        "outputs": {
            "csv": "experiment6_server_accuracy_by_aggregation.csv",
            "markdown": "experiment6_server_accuracy_by_aggregation.md",
            "latex": "experiment6_server_accuracy_by_aggregation.tex",
        },
    }
    write_outputs(table, output_dir, metadata)

    print(f"Wrote Experiment 6 server-accuracy table to: {output_dir}")
    if missing:
        print("Missing algorithm results:")
        for algorithm_key, reason in missing.items():
            print(f"- {algorithm_key}: {reason}")


if __name__ == "__main__":
    main()
