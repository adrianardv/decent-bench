from __future__ import annotations

import argparse
from pathlib import Path

from src.inspection_helpers import (
    InspectionConfig,
    add_seeded_per_writer_train_test_split,
    choose_candidate_clients,
    class_counts,
    client_stats,
    load_huggingface_metadata,
    load_leaf_json_metadata,
    threshold_report,
    write_plots,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect FEMNIST writer/client statistics for thesis experiments.")
    parser.add_argument("--source", choices=["huggingface", "leaf-json"], default="huggingface")
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/femnist/data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/femnist/results/inspection"))
    parser.add_argument("--leaf-train-dir", type=Path)
    parser.add_argument("--leaf-test-dir", type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--candidate-clients", type=int, default=100)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument("--min-test-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "huggingface":
        df = load_huggingface_metadata(args.cache_dir, local_files_only=args.local_files_only)
        df = add_seeded_per_writer_train_test_split(df, train_fraction=args.train_fraction, seed=args.seed)
    else:
        if args.leaf_train_dir is None or args.leaf_test_dir is None:
            raise ValueError("--leaf-train-dir and --leaf-test-dir are required when --source leaf-json is used.")
        df = load_leaf_json_metadata(args.leaf_train_dir, args.leaf_test_dir)

    stats = client_stats(df)
    counts = class_counts(df)
    thresholds = threshold_report(stats)
    selected = choose_candidate_clients(
        stats,
        candidate_clients=args.candidate_clients,
        min_train_samples=args.min_train_samples,
        min_test_samples=args.min_test_samples,
        seed=args.seed,
    )

    stats.to_csv(args.output_dir / "all_clients_stats.csv", index=False)
    counts.to_csv(args.output_dir / "class_counts.csv", index=False)
    thresholds.to_csv(args.output_dir / "client_threshold_report.csv", index=False)
    selected.to_csv(args.output_dir / "selected_clients_stats.csv", index=False)
    write_summary(
        args.output_dir,
        config=InspectionConfig(
            source=args.source,
            seed=args.seed,
            train_fraction=args.train_fraction,
            candidate_clients=args.candidate_clients,
            min_train_samples=args.min_train_samples,
            min_test_samples=args.min_test_samples,
        ),
        df=df,
        stats=stats,
        selected=selected,
    )

    if not args.no_plots:
        write_plots(args.output_dir, stats, counts, selected)

    print(f"Wrote FEMNIST inspection outputs to {args.output_dir}")
    print(f"Rows: {len(df)}")
    print(f"Clients/writers: {stats['writer_id'].nunique()}")
    print(f"Classes: {df['label'].nunique()}")
    print(f"Candidate clients: {len(selected)}")


if __name__ == "__main__":
    main()
