from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import SupportedFrameworks

FEMNIST_CLASS_NAMES = [
    *map(str, range(10)),
    *[chr(code) for code in range(65, 91)],
    *[chr(code) for code in range(97, 123)],
]


class _Figure(Protocol):
    """Subset of Matplotlib's figure API used by the plots."""

    def tight_layout(self) -> None: ...

    def savefig(self, fname: Path, *, dpi: int) -> None: ...

    def colorbar(self, mappable: object, *, ax: _Axes, label: str) -> object: ...


class _Axes(Protocol):
    """Subset of Matplotlib's axes API used by the plots."""

    def hist(self, x: object, *, bins: int, color: str) -> object: ...

    def bar(self, x: object, height: object, *, color: str) -> object: ...

    def imshow(self, x: object, *, aspect: str, cmap: str) -> object: ...

    def set_title(self, label: str) -> object: ...

    def set_xlabel(self, xlabel: str) -> object: ...

    def set_ylabel(self, ylabel: str) -> object: ...

    def tick_params(self, *, axis: str, labelrotation: int) -> object: ...

    def set_xticks(self, ticks: object) -> object: ...

    def set_xticklabels(self, labels: object, rotation: int) -> object: ...

    def set_yticks(self, ticks: object) -> object: ...

    def set_yticklabels(self, labels: object) -> object: ...


class _PyplotModule(Protocol):
    """Subset of Matplotlib's pyplot API used by the plots."""

    def subplots(self, *, figsize: tuple[int, int]) -> tuple[_Figure, _Axes]: ...

    def close(self, fig: _Figure) -> None: ...


@dataclass(frozen=True)
class InspectionConfig:
    """Configuration used to generate FEMNIST inspection outputs."""

    source: str
    seed: int
    train_fraction: float
    candidate_clients: int
    min_train_samples: int
    min_test_samples: int


def load_huggingface_metadata(cache_dir: Path | None, *, local_files_only: bool = False) -> pd.DataFrame:
    """Load lightweight metadata from the Hugging Face FEMNIST dataset."""
    with huggingface_offline_mode(local_files_only):
        try:
            from datasets import (  # type: ignore[import-untyped]  # noqa: PLC0415
                DownloadConfig,
                Image,
                load_dataset,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The Hugging Face FEMNIST source requires the optional 'datasets' package. "
                "Install it with: .venv\\Scripts\\python.exe -m pip install datasets"
            ) from exc

        dataset = load_dataset(
            "flwrlabs/femnist",
            split="train",
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            download_config=DownloadConfig(local_files_only=local_files_only),
        )
    if "image" in dataset.features:
        dataset = dataset.cast_column("image", Image(decode=False))

    metadata_columns = [column for column in ("writer_id", "hsf_id", "character") if column in dataset.column_names]
    if not {"writer_id", "character"}.issubset(metadata_columns):
        raise ValueError(f"Unexpected Hugging Face FEMNIST columns: {dataset.column_names}")

    df = cast("pd.DataFrame", dataset.select_columns(metadata_columns).to_pandas())
    df.insert(0, "row_index", np.arange(len(df), dtype=np.int64))
    df["writer_id"] = df["writer_id"].astype(str)
    df["label"] = df["character"].astype(int)
    return df[["row_index", "writer_id", "label"]]


def load_leaf_json_metadata(train_dir: Path, test_dir: Path) -> pd.DataFrame:
    """Load writer, label, and split metadata from LEAF JSON files."""
    rows: list[dict[str, object]] = []
    for split, directory in (("train", train_dir), ("test", test_dir)):
        if not directory.exists():
            raise FileNotFoundError(f"LEAF {split} directory does not exist: {directory}")
        json_files = sorted(directory.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No LEAF JSON files found in {directory}")
        for json_file in json_files:
            with json_file.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            for user in payload["users"]:
                labels = payload["user_data"][user]["y"]
                for local_index, label in enumerate(labels):
                    rows.append(
                        {
                            "row_index": local_index,
                            "writer_id": str(user),
                            "label": int(label),
                            "split": split,
                            "source_file": json_file.name,
                        }
                    )

    return pd.DataFrame(rows)


def add_seeded_per_writer_train_test_split(df: pd.DataFrame, train_fraction: float, seed: int) -> pd.DataFrame:
    """Add deterministic per-writer train/test labels."""
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    split_df = df.copy()
    rng = _seeded_numpy_rng(seed)
    split_values: np.ndarray = np.empty(len(split_df), dtype=object)

    for _, group in split_df.groupby("writer_id", sort=True):
        positions = group.index.to_numpy()
        shuffled = positions.copy()
        rng.shuffle(shuffled)
        n_train = round(len(shuffled) * train_fraction)
        if len(shuffled) > 1:
            n_train = min(max(n_train, 1), len(shuffled) - 1)
        split_values[shuffled[:n_train]] = "train"
        split_values[shuffled[n_train:]] = "test"

    split_df["split"] = split_values
    return split_df


def client_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-client sample counts, split counts, and label histograms."""
    grouped = df.groupby("writer_id", sort=True)
    stats = grouped.agg(
        total_samples=("label", "size"),
        n_classes=("label", "nunique"),
    )

    split_counts = df.pivot_table(
        index="writer_id", columns="split", values="label", aggfunc="size", fill_value=0
    ).rename(columns={"train": "train_samples", "test": "test_samples"})
    for column in ("train_samples", "test_samples"):
        if column not in split_counts:
            split_counts[column] = 0

    label_histories = grouped["label"].apply(lambda labels: json.dumps(_label_counts(labels)))
    stats = stats.join(split_counts[["train_samples", "test_samples"]], how="left").fillna(0)
    stats["train_samples"] = stats["train_samples"].astype(int)
    stats["test_samples"] = stats["test_samples"].astype(int)
    stats["label_histogram"] = label_histories
    return stats.reset_index()


def class_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return sample counts per FEMNIST class."""
    counts = df.pivot_table(index="label", columns="split", values="writer_id", aggfunc="size", fill_value=0).rename(
        columns={"train": "train_samples", "test": "test_samples"}
    )
    for column in ("train_samples", "test_samples"):
        if column not in counts:
            counts[column] = 0
    counts["total_samples"] = counts["train_samples"] + counts["test_samples"]
    counts = counts.reset_index()[["label", "total_samples", "train_samples", "test_samples"]]
    counts["class_name"] = counts["label"].map(lambda label: FEMNIST_CLASS_NAMES[int(label)])
    return counts[["label", "class_name", "total_samples", "train_samples", "test_samples"]]


def choose_candidate_clients(
    stats: pd.DataFrame,
    *,
    candidate_clients: int,
    min_train_samples: int,
    min_test_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Select eligible clients deterministically from per-client statistics."""
    eligible = stats[(stats["train_samples"] >= min_train_samples) & (stats["test_samples"] >= min_test_samples)].copy()

    if len(eligible) < candidate_clients:
        raise ValueError(
            f"Requested {candidate_clients} clients, but only {len(eligible)} satisfy "
            f"min_train_samples={min_train_samples} and min_test_samples={min_test_samples}."
        )

    rng = _seeded_numpy_rng(seed)
    selected_indices = rng.choice(eligible.index.to_numpy(), size=candidate_clients, replace=False)
    selected = cast("pd.DataFrame", eligible.loc[selected_indices].sort_values("writer_id").reset_index(drop=True))
    selected.insert(0, "client_index", np.arange(len(selected), dtype=np.int64))
    return selected


def _seeded_numpy_rng(seed: int) -> np.random.Generator:
    iop.set_seed(seed, frameworks=[SupportedFrameworks.NUMPY])
    return cast("np.random.Generator", iop.rng_numpy())


def threshold_report(stats: pd.DataFrame) -> pd.DataFrame:
    """Return client eligibility counts for useful train/test thresholds."""
    thresholds = [10, 25, 50, 100, 200, 300, 500]
    rows = []
    for threshold in thresholds:
        eligible = stats[(stats["train_samples"] >= threshold) & (stats["test_samples"] >= max(1, threshold // 5))]
        rows.append(
            {
                "min_train_samples": threshold,
                "min_test_samples": max(1, threshold // 5),
                "eligible_clients": len(eligible),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    output_dir: Path,
    *,
    config: InspectionConfig,
    df: pd.DataFrame,
    stats: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    """Write a JSON summary of the selected FEMNIST inspection configuration."""
    sample_quantiles = stats["total_samples"].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]).to_dict()
    selected_label_totals = _selected_label_totals(selected)
    selected_missing_labels = [
        int(label) for label, total_samples in selected_label_totals.items() if total_samples == 0
    ]
    summary = {
        "source": config.source,
        "seed": config.seed,
        "train_fraction": config.train_fraction,
        "n_rows": len(df),
        "n_clients": int(stats["writer_id"].nunique()),
        "n_classes": int(df["label"].nunique()),
        "train_samples": int((df["split"] == "train").sum()),
        "test_samples": int((df["split"] == "test").sum()),
        "selected_clients": int(config.candidate_clients),
        "selected_min_train_samples": int(config.min_train_samples),
        "selected_min_test_samples": int(config.min_test_samples),
        "selected_total_train_samples": int(selected["train_samples"].sum()),
        "selected_total_test_samples": int(selected["test_samples"].sum()),
        "selected_classes_covered": int(len(FEMNIST_CLASS_NAMES) - len(selected_missing_labels)),
        "selected_missing_classes": selected_missing_labels,
        "samples_per_client_quantiles": {str(key): float(value) for key, value in sample_quantiles.items()},
        "citations": [
            "Caldas et al. (2018), LEAF: A Benchmark for Federated Settings.",
            "Beutel et al. (2020), Flower: A Friendly Federated Learning Research Framework.",
        ],
    }
    with (output_dir / "inspection_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)


def _selected_label_totals(selected: pd.DataFrame) -> dict[int, int]:
    totals = dict.fromkeys(range(len(FEMNIST_CLASS_NAMES)), 0)
    for histogram in selected["label_histogram"]:
        parsed = cast("dict[str, int]", json.loads(histogram))
        for label, count in parsed.items():
            totals[int(label)] += int(count)
    return totals


def write_plots(output_dir: Path, stats: pd.DataFrame, counts: pd.DataFrame, selected: pd.DataFrame, seed: int) -> None:
    """Write inspection plots to disk."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    pyplot = cast("_PyplotModule", plt)
    _plot_samples_per_writer(output_dir, stats, pyplot)
    _plot_samples_per_class(output_dir, counts, pyplot)
    _plot_selected_client_class_distributions(
        output_dir,
        selected,
        pyplot,
        title="Class distribution for selected writers, ordered by writer ID",
        filename="selected_client_class_distributions.png",
    )
    _plot_selected_client_class_distributions(
        output_dir,
        selected.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        pyplot,
        title=f"Class distribution for selected writers, shuffled with seed {seed}",
        filename="selected_client_class_distributions_shuffled.png",
    )


def _label_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _plot_samples_per_writer(output_dir: Path, stats: pd.DataFrame, plt: _PyplotModule) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(stats["total_samples"], bins=50, color="#4C78A8")
    ax.set_title("Samples per writer")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Writers")
    fig.tight_layout()
    fig.savefig(output_dir / "samples_per_writer_histogram.png", dpi=150)
    plt.close(fig)


def _plot_samples_per_class(output_dir: Path, counts: pd.DataFrame, plt: _PyplotModule) -> None:
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(counts["class_name"], counts["total_samples"], color="#F58518")
    ax.set_title("Samples per class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Samples")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / "samples_per_class.png", dpi=150)
    plt.close(fig)


def _plot_selected_client_class_distributions(
    output_dir: Path,
    selected: pd.DataFrame,
    plt: _PyplotModule,
    *,
    title: str,
    filename: str,
) -> None:
    plotted_clients = selected.copy()
    label_counts: np.ndarray = np.zeros((len(plotted_clients), len(FEMNIST_CLASS_NAMES)), dtype=np.int64)

    for row_index, histogram in enumerate(plotted_clients["label_histogram"]):
        parsed = cast("dict[str, int]", json.loads(histogram))
        for label, count in parsed.items():
            label_counts[row_index, int(label)] = int(count)

    figure_height = max(8, min(28, 2 + len(plotted_clients) // 4))
    fig, ax = plt.subplots(figsize=(18, figure_height))
    image = ax.imshow(label_counts, aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Writer ID")
    ax.set_xticks(np.arange(len(FEMNIST_CLASS_NAMES)))
    ax.set_xticklabels(FEMNIST_CLASS_NAMES, rotation=90)
    ax.set_yticks(np.arange(len(plotted_clients)))
    ax.set_yticklabels(plotted_clients["writer_id"].astype(str))
    fig.colorbar(image, ax=ax, label="Samples")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


@contextmanager
def huggingface_offline_mode(enabled: bool) -> Iterator[None]:
    """Temporarily enable Hugging Face offline mode."""
    previous_value = os.environ.get("HF_HUB_OFFLINE")
    if enabled:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if enabled:
            if previous_value is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_value
