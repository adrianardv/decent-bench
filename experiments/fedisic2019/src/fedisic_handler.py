from __future__ import annotations

import os
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from decent_bench.datasets import DatasetHandler
from decent_bench.utils.types import Datapoint

from .transforms import IMAGE_SIZE, build_test_transform, build_train_transform, image_to_chw_float32


SplitName = Literal["train", "test"]
SampleStrategy = Literal["head", "stratified"]
ISIC_CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CENTER_NAMES = {
    0: "BCN",
    1: "HAM_vidir_molemax",
    2: "HAM_vidir_modern",
    3: "HAM_rosendahl",
    4: "MSK",
    5: "HAM_vienna_dias",
}


class _HuggingFaceDataset(Protocol):
    features: Any
    column_names: list[str]

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict[str, object]: ...


class FedISICDatasetHandler(DatasetHandler):
    """
    Fed-ISIC2019 dataset handler using natural center/site partitions.

    The default source is the Flower Labs Hugging Face copy of Fed-ISIC2019.
    This source mirrors FLamby's six-center split and stores RGB images,
    center IDs, integer labels, and train/test splits. Runtime transforms are
    applied by this handler.
    """

    def __init__(
        self,
        *,
        split: SplitName,
        dataset_name: str = "flwrlabs/fed-isic2019",
        cache_dir: Path | str | None = Path("experiments/fedisic2019/data/cache"),
        centers: Sequence[int] | None = None,
        image_size: int = IMAGE_SIZE,
        max_samples_per_client: int | None = None,
        sample_fraction_per_client: float | None = None,
        min_samples_per_client: int | None = None,
        sample_strategy: SampleStrategy = "head",
        seed: int = 20260524,
        local_files_only: bool = False,
    ) -> None:
        _validate_init_args(
            split,
            image_size,
            max_samples_per_client,
            sample_fraction_per_client,
            min_samples_per_client,
            sample_strategy,
        )
        self.split = split
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.requested_centers = list(centers) if centers is not None else None
        self.image_size = image_size
        self.max_samples_per_client = max_samples_per_client
        self.sample_fraction_per_client = sample_fraction_per_client
        self.min_samples_per_client = min_samples_per_client
        self.sample_strategy = sample_strategy
        self.seed = int(seed)
        self.local_files_only = local_files_only

        self._partitions: list[FedISICPartition] | None = None

    @property
    def n_samples(self) -> int:
        """Return the number of datapoints in the selected split and centers."""
        return sum(len(partition) for partition in self.get_partitions())

    @property
    def n_partitions(self) -> int:
        """Return the number of selected Fed-ISIC2019 centers."""
        return len(self.center_ids)

    @property
    def n_features(self) -> int:
        """Return the flattened C x H x W image feature count."""
        return 3 * self.image_size * self.image_size

    @property
    def n_targets(self) -> int:
        """Return the number of label classes found in metadata."""
        return int(self.metadata[self.label_column].nunique())

    @cached_property
    def center_ids(self) -> list[int]:
        """Return selected center IDs after validating them against the source metadata."""
        available = sorted(int(center) for center in self.metadata[self.center_column].unique())
        if self.requested_centers is None:
            return available
        requested = [int(center) for center in self.requested_centers]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested centers {missing} are not available. Available centers: {available}")
        return requested

    @cached_property
    def class_names(self) -> list[str]:
        """Return readable class names from the dataset feature metadata, or ISIC defaults."""
        label_feature = self.hf_dataset.features.get(self.label_column)
        feature_names = getattr(label_feature, "names", None)
        if (
            isinstance(feature_names, list)
            and len(feature_names) == self.n_targets
            and not all(str(name).isdigit() for name in feature_names)
        ):
            return [str(name) for name in feature_names]
        if self.n_targets == len(ISIC_CLASS_NAMES):
            return ISIC_CLASS_NAMES
        return [str(label) for label in sorted(self.metadata[self.label_column].unique())]

    @cached_property
    def center_names(self) -> dict[int, str]:
        """Return readable center names where the FLamby center order is known."""
        return {center: CENTER_NAMES.get(center, f"center_{center}") for center in self.center_ids}

    @cached_property
    def metadata(self) -> pd.DataFrame:
        """Return split metadata with row indices, center IDs, and integer labels."""
        dataset = self._load_hf_dataset(decode_images=False)
        center_column = _first_existing_column(dataset.column_names, ("center", "site", "client_id", "partition_id"))
        label_column = _first_existing_column(dataset.column_names, ("label", "target"))
        columns = [center_column, label_column]
        df = cast("pd.DataFrame", dataset.select_columns(columns).to_pandas())
        df.insert(0, "row_index", np.arange(len(df), dtype=np.int64))
        df[center_column] = df[center_column].astype(int)
        df[label_column] = df[label_column].astype(int)
        return df[["row_index", center_column, label_column]]

    @cached_property
    def center_column(self) -> str:
        """Return the source column holding center/site IDs."""
        return _first_existing_column(self.metadata.columns, ("center", "site", "client_id", "partition_id"))

    @cached_property
    def label_column(self) -> str:
        """Return the source column holding integer labels."""
        return _first_existing_column(self.metadata.columns, ("label", "target"))

    @cached_property
    def hf_dataset(self) -> _HuggingFaceDataset:
        """Return the image-decoding Hugging Face dataset split."""
        return self._load_hf_dataset(decode_images=True)

    def get_datapoints(self) -> FedISICPooledDataset:
        """Return the selected split as one lazy pooled dataset."""
        return FedISICPooledDataset(self.get_partitions())

    def get_partitions(self) -> list["FedISICPartition"]:
        """Return one lazy dataset partition per selected Fed-ISIC2019 center."""
        if self._partitions is None:
            transform = build_train_transform(self.image_size) if self.split == "train" else build_test_transform(self.image_size)
            self._partitions = [
                self._build_center_partition(center_id, transform=transform) for center_id in self.center_ids
            ]
        return self._partitions

    def _build_center_partition(self, center_id: int, *, transform: Any) -> "FedISICPartition":
        rows = self.metadata[self.metadata[self.center_column] == center_id].sort_values("row_index")
        target_samples = _target_samples_for_partition(
            n_available=len(rows),
            max_samples_per_client=self.max_samples_per_client,
            sample_fraction_per_client=self.sample_fraction_per_client,
            min_samples_per_client=self.min_samples_per_client,
        )
        if target_samples is not None:
            rows = _sample_rows_for_partition(
                rows,
                label_column=self.label_column,
                max_samples=target_samples,
                strategy=self.sample_strategy,
                seed=self.seed + (100_000 if self.split == "test" else 0) + int(center_id) * 1009,
            )
        return FedISICPartition(
            hf_dataset=self.hf_dataset,
            row_indices=[int(row_index) for row_index in rows["row_index"].to_list()],
            labels=[int(label) for label in rows[self.label_column].to_list()],
            center_id=int(center_id),
            center_name=CENTER_NAMES.get(int(center_id), f"center_{center_id}"),
            label_column=self.label_column,
            transform=transform,
        )

    def _load_hf_dataset(self, *, decode_images: bool) -> _HuggingFaceDataset:
        with huggingface_offline_mode(self.local_files_only):
            try:
                from datasets import DownloadConfig, Image, load_dataset  # type: ignore[import-untyped]  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "FedISICDatasetHandler requires the optional 'datasets' package. "
                    "Install it with: .venv\\Scripts\\python.exe -m pip install -e .[dev]"
                ) from exc

            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                download_config=DownloadConfig(local_files_only=self.local_files_only),
            )
        if not decode_images and "image" in dataset.features:
            dataset = dataset.cast_column("image", Image(decode=False))
        return cast("_HuggingFaceDataset", dataset)


class FedISICPartition(Sequence[Datapoint]):
    """Lazy Fed-ISIC2019 center partition that preprocesses images on access."""

    def __init__(
        self,
        *,
        hf_dataset: _HuggingFaceDataset,
        row_indices: Sequence[int],
        labels: Sequence[int],
        center_id: int,
        center_name: str,
        label_column: str,
        transform: Any,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.row_indices = list(row_indices)
        self.labels = list(labels)
        self.center_id = center_id
        self.center_name = center_name
        self.label_column = label_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> Datapoint:
        import torch  # noqa: PLC0415

        row = self.hf_dataset[self.row_indices[index]]
        image = image_to_chw_float32(row["image"], self.transform)
        label = int(row[self.label_column])
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


class FedISICPooledDataset(Sequence[Datapoint]):
    """Lazy concatenation of Fed-ISIC2019 center partitions."""

    def __init__(self, partitions: Sequence[FedISICPartition]) -> None:
        self.partitions = list(partitions)
        self._cumulative_lengths = np.cumsum([len(partition) for partition in self.partitions]).tolist()

    def __len__(self) -> int:
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0

    def __getitem__(self, index: int) -> Datapoint:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        partition_index = bisect_right(self._cumulative_lengths, index)
        previous_length = 0 if partition_index == 0 else self._cumulative_lengths[partition_index - 1]
        return self.partitions[partition_index][index - previous_length]


def _validate_init_args(
    split: SplitName,
    image_size: int,
    max_samples_per_client: int | None,
    sample_fraction_per_client: float | None,
    min_samples_per_client: int | None,
    sample_strategy: SampleStrategy,
) -> None:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if max_samples_per_client is not None and max_samples_per_client <= 0:
        raise ValueError(f"max_samples_per_client must be positive, got {max_samples_per_client}")
    if sample_fraction_per_client is not None and not 0 < sample_fraction_per_client <= 1:
        raise ValueError(f"sample_fraction_per_client must be in (0, 1], got {sample_fraction_per_client}")
    if min_samples_per_client is not None and min_samples_per_client <= 0:
        raise ValueError(f"min_samples_per_client must be positive, got {min_samples_per_client}")
    if max_samples_per_client is not None and sample_fraction_per_client is not None:
        raise ValueError("max_samples_per_client and sample_fraction_per_client cannot both be set")
    if sample_strategy not in ("head", "stratified"):
        raise ValueError(f"sample_strategy must be 'head' or 'stratified', got {sample_strategy!r}")


def _target_samples_for_partition(
    *,
    n_available: int,
    max_samples_per_client: int | None,
    sample_fraction_per_client: float | None,
    min_samples_per_client: int | None,
) -> int | None:
    if max_samples_per_client is not None:
        return min(max_samples_per_client, n_available)
    if sample_fraction_per_client is None and min_samples_per_client is None:
        return None

    target = n_available if sample_fraction_per_client is None else int(np.ceil(n_available * sample_fraction_per_client))
    if min_samples_per_client is not None:
        target = max(target, min_samples_per_client)
    return min(target, n_available)


def _sample_rows_for_partition(
    rows: pd.DataFrame,
    *,
    label_column: str,
    max_samples: int,
    strategy: SampleStrategy,
    seed: int,
) -> pd.DataFrame:
    if len(rows) <= max_samples:
        return rows.sort_values("row_index")
    if strategy == "head":
        return rows.head(max_samples)
    return _stratified_sample_rows(rows, label_column=label_column, max_samples=max_samples, seed=seed)


def _stratified_sample_rows(
    rows: pd.DataFrame,
    *,
    label_column: str,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Return a deterministic per-class proportional sample from one center."""
    counts = rows[label_column].value_counts().sort_index()
    if counts.empty:
        return rows.head(0)

    quotas = pd.Series(0, index=counts.index, dtype=int)
    if max_samples >= len(counts):
        quotas += 1
        remaining = max_samples - int(quotas.sum())
    else:
        remaining = max_samples

    raw = counts / counts.sum() * remaining
    additions = np.floor(raw).astype(int).clip(upper=counts - quotas)
    quotas += additions

    while int(quotas.sum()) < max_samples:
        capacity = counts - quotas
        available = capacity[capacity > 0]
        if available.empty:
            break
        remainders = (raw - np.floor(raw)).reindex(available.index).fillna(0.0)
        next_label = sorted(
            available.index,
            key=lambda label: (float(remainders.loc[label]), int(available.loc[label]), -int(label)),
            reverse=True,
        )[0]
        quotas.loc[next_label] += 1

    while int(quotas.sum()) > max_samples:
        removable = quotas[quotas > 1]
        if removable.empty:
            removable = quotas[quotas > 0]
        next_label = sorted(
            removable.index,
            key=lambda label: (float(quotas.loc[label]), -float(counts.loc[label]), -int(label)),
            reverse=True,
        )[0]
        quotas.loc[next_label] -= 1

    rng = np.random.default_rng(seed)
    sampled_frames = []
    for label, quota in quotas.items():
        if int(quota) <= 0:
            continue
        class_rows = rows[rows[label_column] == label]
        selected_positions = rng.choice(len(class_rows), size=int(quota), replace=False)
        sampled_frames.append(class_rows.iloc[np.sort(selected_positions)])

    if not sampled_frames:
        return rows.head(0)
    return pd.concat(sampled_frames, ignore_index=False).sort_values("row_index")


def _first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"None of columns {candidates!r} were found. Available columns: {list(columns)!r}")


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
