# ruff: noqa: ANN401, D103, INP001, T201
"""
Post-process Experiment 5 metric files and compute drops from clean baseline.

The script compares each impairment condition against the clean baseline for
each algorithm. It defaults to the latest run found under each condition
directory.
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

checkpoint_root = Path("experiments/femnist/checkpoints/experiment5")
default_output_root = Path("experiments/femnist/results/experiment5/performance_drop")
metric_result_filename = "metric_computation.pkl.zst"
baseline_condition = "clean_baseline"

condition_order = (
    "clean_baseline",
    "activation_uniform_low",
    "activation_uniform_high",
    "activation_markov_high_availability",
    "activation_markov_low_availability",
    "compression_topk_low",
    "compression_topk_high",
    "compression_qsgd_low",
    "compression_qsgd_high",
    "drops_uniform_low",
    "drops_uniform_high",
    "noise_gaussian_low",
    "noise_gaussian_high",
    "combined_uniform_topk_drops",
)

headline_metrics = {
    ("server accuracy", ""),
    ("accuracy", "avg"),
}

higher_is_better_metrics = {
    "accuracy",
    "fraction selected clients",
    "server accuracy",
}

lower_is_better_metrics = {
    "client drift from server",
    "loss",
    "nr gradient calls",
    "nr received messages",
    "nr sent messages",
    "nr sent messages dropped",
}


def latest_metric_file(condition: str, root: Path) -> Path | None:
    condition_dir = root / condition
    candidates = sorted(condition_dir.glob(f"run_*/{metric_result_filename}"))
    return candidates[-1] if candidates else None


def load_metric_result(path: Path) -> Any:
    decompressed = zstd.ZstdDecompressor().decompress(path.read_bytes())
    return pickle.loads(decompressed)  # noqa: S301


def load_table(path: Path, condition: str) -> pd.DataFrame:
    metric_result = load_metric_result(path)
    table_df, _ = metric_result.to_dataframe()
    if table_df is None:
        raise ValueError(f"No table metrics found in {path}")
    table_df = table_df.copy()
    table_df.insert(0, "condition", condition)
    table_df.insert(1, "metric_file", str(path))
    return table_df


def collect_tables(root: Path, requested_conditions: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    tables: list[pd.DataFrame] = []
    missing: dict[str, str] = {}
    for condition in requested_conditions:
        metric_file = latest_metric_file(condition, root)
        if metric_file is None:
            missing[condition] = f"No {metric_result_filename} found under {root / condition}"
            continue
        tables.append(load_table(metric_file, condition))

    if not tables:
        raise RuntimeError(f"No Experiment 5 metric files found under {root}")
    return pd.concat(tables, ignore_index=True), missing


def metric_direction(metric: str) -> str:
    if metric in higher_is_better_metrics:
        return "higher_is_better"
    if metric in lower_is_better_metrics:
        return "lower_is_better"
    return "unknown"


def relative_change_pct(baseline: float, condition: float) -> float:
    if baseline == 0:
        return float("nan")
    return ((condition - baseline) / abs(baseline)) * 100.0


def degradation_pct(metric: str, baseline: float, condition: float) -> float:
    if baseline == 0:
        return float("nan")
    if metric_direction(metric) == "higher_is_better":
        return ((baseline - condition) / abs(baseline)) * 100.0
    if metric_direction(metric) == "lower_is_better":
        return ((condition - baseline) / abs(baseline)) * 100.0
    return float("nan")


def build_comparison_table(table_df: pd.DataFrame) -> pd.DataFrame:
    baseline_df = table_df[table_df["condition"] == baseline_condition]
    if baseline_df.empty:
        raise RuntimeError(f"Missing required baseline condition: {baseline_condition}")

    baseline_values = baseline_df[
        ["algorithm", "metric", "statistic", "mean", "margin_of_error", "metric_file"]
    ].rename(
        columns={
            "mean": "baseline_mean",
            "margin_of_error": "baseline_margin_of_error",
            "metric_file": "baseline_metric_file",
        }
    )
    condition_values = table_df[table_df["condition"] != baseline_condition].rename(
        columns={
            "mean": "condition_mean",
            "margin_of_error": "condition_margin_of_error",
            "metric_file": "condition_metric_file",
        }
    )

    comparison = condition_values.merge(
        baseline_values,
        on=["algorithm", "metric", "statistic"],
        how="inner",
    )
    comparison["relative_change_pct"] = comparison.apply(
        lambda row: relative_change_pct(row["baseline_mean"], row["condition_mean"]),
        axis=1,
    )
    comparison["degradation_pct"] = comparison.apply(
        lambda row: degradation_pct(row["metric"], row["baseline_mean"], row["condition_mean"]),
        axis=1,
    )
    comparison["metric_direction"] = comparison["metric"].map(metric_direction)
    return comparison[
        [
            "condition",
            "algorithm",
            "metric",
            "statistic",
            "metric_direction",
            "baseline_mean",
            "condition_mean",
            "relative_change_pct",
            "degradation_pct",
            "baseline_margin_of_error",
            "condition_margin_of_error",
            "baseline_metric_file",
            "condition_metric_file",
        ]
    ].sort_values(["condition", "algorithm", "metric", "statistic"])


def build_headline_drop_table(comparison: pd.DataFrame) -> pd.DataFrame:
    headline = comparison[
        comparison.apply(lambda row: (row["metric"], row["statistic"]) in headline_metrics, axis=1)
    ].copy()
    headline = headline.rename(columns={"degradation_pct": "performance_drop_pct"})
    return headline[
        [
            "condition",
            "algorithm",
            "metric",
            "statistic",
            "baseline_mean",
            "condition_mean",
            "performance_drop_pct",
            "baseline_margin_of_error",
            "condition_margin_of_error",
            "baseline_metric_file",
            "condition_metric_file",
        ]
    ].sort_values(["condition", "algorithm", "metric", "statistic"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=checkpoint_root,
        help="Root containing Experiment 5 condition run folders.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(condition_order),
        choices=list(condition_order),
        help="Conditions to include. clean_baseline is required for comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV/JSON outputs. Defaults to a timestamped folder under experiments/femnist/results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_conditions = list(dict.fromkeys([baseline_condition, *args.conditions]))
    output_dir = args.output_dir or default_output_root / f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    table_df, missing = collect_tables(args.checkpoint_root, requested_conditions)
    comparison = build_comparison_table(table_df)
    headline = build_headline_drop_table(comparison)

    table_df.to_csv(output_dir / "experiment5_table_metrics.csv", index=False)
    comparison.to_csv(output_dir / "experiment5_metric_relative_changes.csv", index=False)
    headline.to_csv(output_dir / "experiment5_performance_drop_from_clean_baseline.csv", index=False)
    (output_dir / "experiment5_postprocess_metadata.json").write_text(
        json.dumps(
            {
                "baseline_condition": baseline_condition,
                "checkpoint_root": str(args.checkpoint_root),
                "requested_conditions": requested_conditions,
                "missing_conditions": missing,
                "headline_metrics": sorted([f"{metric}:{statistic}" for metric, statistic in headline_metrics]),
                "outputs": {
                    "table_metrics": "experiment5_table_metrics.csv",
                    "all_relative_changes": "experiment5_metric_relative_changes.csv",
                    "headline_performance_drop": "experiment5_performance_drop_from_clean_baseline.csv",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote Experiment 5 post-processing outputs to: {output_dir}")
    if missing:
        print("Missing condition results:")
        for condition, reason in missing.items():
            print(f"- {condition}: {reason}")


if __name__ == "__main__":
    main()
