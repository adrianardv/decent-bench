from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from .fedisic_handler import FedISICDatasetHandler


def load_full_metadata(
    *,
    cache_dir: Path | None,
    local_files_only: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[int, str]]:
    """Load normalized train+test metadata for the full Fed-ISIC2019 dataset."""
    frames = []
    class_names: list[str] | None = None
    center_names: dict[int, str] | None = None
    for split in ("train", "test"):
        handler = FedISICDatasetHandler(split=split, cache_dir=cache_dir, local_files_only=local_files_only)
        frame = handler.metadata.rename(columns={handler.center_column: "center", handler.label_column: "label"})
        frame = frame.copy()
        frame["split"] = split
        frames.append(frame[["split", "row_index", "center", "label"]])
        class_names = handler.class_names
        center_names = handler.center_names

    metadata = pd.concat(frames, ignore_index=True)
    return metadata, cast("list[str]", class_names), cast("dict[int, str]", center_names)


def write_inspection_outputs(
    output_dir: Path,
    *,
    cache_dir: Path | None,
    local_files_only: bool = False,
    seed: int = 20260524,
) -> None:
    """Write Fed-ISIC2019 metadata summaries and inspection figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, class_names, center_names = load_full_metadata(cache_dir=cache_dir, local_files_only=local_files_only)
    class_counts = _class_counts(metadata, class_names)
    center_counts = _center_counts(metadata, center_names)
    center_class_counts = _center_class_counts(metadata, class_names, center_names)
    image_sizes = _image_sizes(cache_dir=cache_dir, local_files_only=local_files_only)

    metadata.to_csv(output_dir / "metadata.csv", index=False)
    class_counts.to_csv(output_dir / "class_counts.csv", index=False)
    center_counts.to_csv(output_dir / "client_counts.csv", index=False)
    center_class_counts.to_csv(output_dir / "client_class_counts.csv", index=False)
    image_sizes.to_csv(output_dir / "image_size_counts.csv", index=False)
    _write_summary(output_dir, metadata, class_counts, center_counts, image_sizes)
    _write_plots(output_dir, metadata, class_counts, center_counts, center_class_counts, class_names, seed, cache_dir, local_files_only)


def _class_counts(metadata: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    counts = metadata.pivot_table(index="label", columns="split", values="row_index", aggfunc="size", fill_value=0)
    for column in ("train", "test"):
        if column not in counts:
            counts[column] = 0
    counts["total"] = counts["train"] + counts["test"]
    counts = counts.reset_index()
    counts["class_name"] = counts["label"].map(lambda label: class_names[int(label)] if int(label) < len(class_names) else str(label))
    return counts[["label", "class_name", "total", "train", "test"]]


def _center_counts(metadata: pd.DataFrame, center_names: dict[int, str]) -> pd.DataFrame:
    counts = metadata.pivot_table(index="center", columns="split", values="row_index", aggfunc="size", fill_value=0)
    for column in ("train", "test"):
        if column not in counts:
            counts[column] = 0
    counts["total"] = counts["train"] + counts["test"]
    counts = counts.reset_index()
    counts["center_name"] = counts["center"].map(lambda center: center_names.get(int(center), f"center_{center}"))
    return counts[["center", "center_name", "total", "train", "test"]]


def _center_class_counts(metadata: pd.DataFrame, class_names: list[str], center_names: dict[int, str]) -> pd.DataFrame:
    counts = metadata.pivot_table(index="center", columns="label", values="row_index", aggfunc="size", fill_value=0)
    counts = counts.reindex(columns=range(len(class_names)), fill_value=0)
    counts = counts.reset_index()
    counts.insert(1, "center_name", counts["center"].map(lambda center: center_names.get(int(center), f"center_{center}")))
    return counts


def _write_summary(
    output_dir: Path,
    metadata: pd.DataFrame,
    class_counts: pd.DataFrame,
    center_counts: pd.DataFrame,
    image_sizes: pd.DataFrame,
) -> None:
    min_short_edge = int(image_sizes[["width", "height"]].min(axis=1).min()) if not image_sizes.empty else None
    max_short_edge = int(image_sizes[["width", "height"]].min(axis=1).max()) if not image_sizes.empty else None
    min_width = int(image_sizes["width"].min()) if not image_sizes.empty else None
    min_height = int(image_sizes["height"].min()) if not image_sizes.empty else None
    summary = {
        "dataset": "Fed-ISIC2019",
        "source": "flwrlabs/fed-isic2019",
        "n_rows": int(len(metadata)),
        "n_train": int((metadata["split"] == "train").sum()),
        "n_test": int((metadata["split"] == "test").sum()),
        "n_centers": int(metadata["center"].nunique()),
        "n_classes": int(metadata["label"].nunique()),
        "image_size_unique_count": int(len(image_sizes)),
        "image_min_width": min_width,
        "image_min_height": min_height,
        "image_min_short_edge": min_short_edge,
        "image_max_short_edge": max_short_edge,
        "images_safe_for_200_crop": bool(min_short_edge is not None and min_short_edge >= 200),
        "class_counts": class_counts.to_dict(orient="records"),
        "center_counts": center_counts.to_dict(orient="records"),
        "top_image_sizes": image_sizes.head(20).to_dict(orient="records"),
    }
    (output_dir / "inspection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _image_sizes(*, cache_dir: Path | None, local_files_only: bool) -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        handler = FedISICDatasetHandler(split=split, cache_dir=cache_dir, local_files_only=local_files_only)
        for row_index in range(len(handler.hf_dataset)):
            image = handler.hf_dataset[row_index]["image"]
            if hasattr(image, "size") and not isinstance(image.size, int):
                width, height = image.size
            else:
                image_array = np.asarray(image)
                width, height = image_array.shape[1], image_array.shape[0]
            rows.append({"split": split, "width": int(width), "height": int(height)})
    frame = pd.DataFrame(rows)
    counts = frame.value_counts(["split", "width", "height"]).reset_index(name="samples")
    return counts.sort_values(["split", "samples"], ascending=[True, False]).reset_index(drop=True)


def _write_plots(
    output_dir: Path,
    metadata: pd.DataFrame,
    class_counts: pd.DataFrame,
    center_counts: pd.DataFrame,
    center_class_counts: pd.DataFrame,
    class_names: list[str],
    seed: int,
    cache_dir: Path | None,
    local_files_only: bool,
) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    _plot_example_grid(output_dir, metadata, class_names, seed, cache_dir, local_files_only, plt)
    _plot_samples_per_class(output_dir, class_counts, plt)
    _plot_samples_per_client(output_dir, center_counts, plt)
    _plot_client_class_distribution(output_dir, center_class_counts, class_names, plt)


def _plot_example_grid(
    output_dir: Path,
    metadata: pd.DataFrame,
    class_names: list[str],
    seed: int,
    cache_dir: Path | None,
    local_files_only: bool,
    plt: object,
) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for label, group in metadata.groupby("label", sort=True):
        selected = group.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
        rows.append(selected.iloc[0])

    split_handlers = {
        split: FedISICDatasetHandler(split=split, cache_dir=cache_dir, local_files_only=local_files_only)
        for split in ("train", "test")
    }
    n_cols = 4
    n_rows = int(np.ceil(len(rows) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    flat_axes = np.asarray(axes).reshape(-1)
    for ax, row in zip(flat_axes, rows, strict=False):
        handler = split_handlers[str(row["split"])]
        example = handler.hf_dataset[int(row["row_index"])]
        image = example["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")  # type: ignore[attr-defined]
        ax.imshow(image)
        label = int(row["label"])
        ax.set_title(class_names[label] if label < len(class_names) else str(label))
        ax.axis("off")
    for ax in flat_axes[len(rows) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "example_grid.png", dpi=150)
    plt.close(fig)


def _plot_samples_per_class(output_dir: Path, class_counts: pd.DataFrame, plt: object) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(class_counts["class_name"], class_counts["total"], color="#4C78A8")
    ax.set_title("Fed-ISIC2019 samples per class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Samples")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "samples_per_class.png", dpi=150)
    plt.close(fig)


def _plot_samples_per_client(output_dir: Path, center_counts: pd.DataFrame, plt: object) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(center_counts["center_name"], center_counts["total"], color="#F58518")
    ax.set_title("Fed-ISIC2019 samples per client/site")
    ax.set_xlabel("Client/site")
    ax.set_ylabel("Samples")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "samples_per_client.png", dpi=150)
    plt.close(fig)


def _plot_client_class_distribution(
    output_dir: Path,
    center_class_counts: pd.DataFrame,
    class_names: list[str],
    plt: object,
) -> None:
    values = center_class_counts[list(range(len(class_names)))].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(values, aspect="auto", cmap="Blues")
    ax.set_title("Fed-ISIC2019 client-by-class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client/site")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(np.arange(len(center_class_counts)))
    ax.set_yticklabels(center_class_counts["center_name"])
    fig.colorbar(image, ax=ax, label="Samples")
    fig.tight_layout()
    fig.savefig(output_dir / "client_class_distribution.png", dpi=150)
    plt.close(fig)
