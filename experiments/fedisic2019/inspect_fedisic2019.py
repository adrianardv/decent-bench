from __future__ import annotations

import argparse
from pathlib import Path

from src.inspection_helpers import write_inspection_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Fed-ISIC2019 center and class statistics.")
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/fedisic2019/data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/fedisic2019/figures"))
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_inspection_outputs(
        args.output_dir,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        seed=args.seed,
    )
    print(f"Wrote Fed-ISIC2019 inspection outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
