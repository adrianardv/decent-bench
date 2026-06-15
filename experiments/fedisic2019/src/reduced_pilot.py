from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .fedisic_handler import FedISICDatasetHandler, FedISICPartition


DEFAULT_REDUCED_SAMPLE_FRACTION_PER_CENTER = 0.10
DEFAULT_REDUCED_MIN_SAMPLES_PER_CENTER = 100


def build_reduced_handlers(
    *,
    sample_fraction_per_center: float = DEFAULT_REDUCED_SAMPLE_FRACTION_PER_CENTER,
    min_samples_per_center: int = DEFAULT_REDUCED_MIN_SAMPLES_PER_CENTER,
    seed: int = 20260524,
    cache_dir: Path | str | None = Path("experiments/fedisic2019/data/cache"),
    local_files_only: bool = False,
) -> tuple[FedISICDatasetHandler, FedISICDatasetHandler]:
    """Build deterministic stratified capped Fed-ISIC2019 train/test handlers."""
    train_dataset = FedISICDatasetHandler(
        split="train",
        cache_dir=cache_dir,
        sample_fraction_per_client=sample_fraction_per_center,
        min_samples_per_client=min_samples_per_center,
        sample_strategy="stratified",
        seed=seed,
        local_files_only=local_files_only,
    )
    test_dataset = FedISICDatasetHandler(
        split="test",
        cache_dir=cache_dir,
        sample_fraction_per_client=sample_fraction_per_center,
        min_samples_per_client=min_samples_per_center,
        sample_strategy="stratified",
        seed=seed,
        local_files_only=local_files_only,
    )
    return train_dataset, test_dataset


def write_reduced_distribution_outputs(
    *,
    output_dir: Path,
    train_dataset: FedISICDatasetHandler,
    test_dataset: FedISICDatasetHandler,
    title_suffix: str = "reduced pilot",
) -> dict[str, Any]:
    """Write reduced-dataset class distribution CSVs and a class-count plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = reduced_partition_metadata(
        train_partitions=train_dataset.get_partitions(),
        test_partitions=test_dataset.get_partitions(),
        class_names=train_dataset.class_names,
    )
    class_counts = _class_counts(metadata)
    center_class_counts = _center_class_counts(metadata)

    metadata_path = output_dir / "reduced_dataset_metadata.csv"
    class_counts_path = output_dir / "reduced_dataset_class_counts.csv"
    center_class_counts_path = output_dir / "reduced_dataset_center_class_counts.csv"
    plot_path = output_dir / "reduced_dataset_class_distribution.png"

    metadata.to_csv(metadata_path, index=False)
    class_counts.to_csv(class_counts_path, index=False)
    center_class_counts.to_csv(center_class_counts_path, index=False)
    _plot_class_distribution(class_counts, plot_path, title_suffix)

    return {
        "metadata_csv": str(metadata_path),
        "class_counts_csv": str(class_counts_path),
        "center_class_counts_csv": str(center_class_counts_path),
        "class_distribution_plot": str(plot_path),
        "n_train_samples": int((metadata["split"] == "train").sum()),
        "n_test_samples": int((metadata["split"] == "test").sum()),
        "train_samples_per_center": _split_center_counts(metadata, "train"),
        "test_samples_per_center": _split_center_counts(metadata, "test"),
        "class_counts": class_counts.to_dict(orient="records"),
    }


def reduced_partition_metadata(
    *,
    train_partitions: list[FedISICPartition],
    test_partitions: list[FedISICPartition],
    class_names: list[str],
) -> pd.DataFrame:
    frames = [
        _partition_metadata(train_partitions, split="train", class_names=class_names),
        _partition_metadata(test_partitions, split="test", class_names=class_names),
    ]
    return pd.concat(frames, ignore_index=True)


def _partition_metadata(
    partitions: list[FedISICPartition],
    *,
    split: str,
    class_names: list[str],
) -> pd.DataFrame:
    rows = []
    for partition in partitions:
        for row_index, label in zip(partition.row_indices, partition.labels, strict=True):
            class_name = class_names[int(label)] if int(label) < len(class_names) else str(label)
            rows.append(
                {
                    "split": split,
                    "center_id": int(partition.center_id),
                    "center_name": partition.center_name,
                    "row_index": int(row_index),
                    "label": int(label),
                    "class_name": class_name,
                }
            )
    return pd.DataFrame(rows)


def _class_counts(metadata: pd.DataFrame) -> pd.DataFrame:
    counts = metadata.pivot_table(index=["label", "class_name"], columns="split", values="row_index", aggfunc="size", fill_value=0)
    for split in ("train", "test"):
        if split not in counts:
            counts[split] = 0
    counts["total"] = counts["train"] + counts["test"]
    return counts.reset_index()[["label", "class_name", "total", "train", "test"]]


def _center_class_counts(metadata: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata.groupby(["split", "center_id", "center_name", "label", "class_name"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(["split", "center_id", "label"])
        .reset_index(drop=True)
    )


def _split_center_counts(metadata: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    split_metadata = metadata[metadata["split"] == split]
    counts = (
        split_metadata.groupby(["center_id", "center_name"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
        .sort_values("center_id")
    )
    return counts.to_dict(orient="records")


def _plot_class_distribution(class_counts: pd.DataFrame, plot_path: Path, title_suffix: str) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    x = np.arange(len(class_counts))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, class_counts["train"], width=width, label="train", color="#4C78A8")
    ax.bar(x + width / 2, class_counts["test"], width=width, label="test", color="#F58518")
    ax.set_title(f"Fed-ISIC2019 class distribution ({title_suffix})")
    ax.set_xlabel("Class")
    ax.set_ylabel("Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(class_counts["class_name"], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
