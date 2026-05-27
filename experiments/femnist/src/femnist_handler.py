from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
import pandas as pd

from decent_bench.datasets import DatasetHandler
from decent_bench.utils.types import Datapoint, Dataset

from .inspection_helpers import (
    add_seeded_per_writer_train_test_split,
    choose_candidate_clients,
    client_stats,
    huggingface_offline_mode,
    load_huggingface_metadata,
)

if TYPE_CHECKING:
    import torch

SplitName = Literal["train", "test"]
ImageLayout = Literal["cnn", "flat"]


class _HuggingFaceDataset(Protocol):
    """Subset of the Hugging Face dataset API used by this handler."""

    def __getitem__(self, index: int) -> dict[str, object]: ...


class _ConvertibleImage(Protocol):
    """Image-like object with PIL's ``convert`` method."""

    def convert(self, mode: str) -> object: ...


class FEMNISTDatasetHandler(DatasetHandler):
    """
    FEMNIST dataset handler using natural writer/client partitions.

    FEMNIST was introduced through LEAF. This experiment handler currently reads
    the Flower Labs Hugging Face copy and keeps LEAF's natural writer-based
    client structure. See ``experiments/femnist/references.bib`` for citations.

    The handler follows the same usage pattern as the MNIST example: create one
    handler for the train split and one for the test split. Calling
    ``get_partitions`` on the train handler returns one local dataset per
    selected writer. Calling ``get_datapoints`` on the test handler returns the
    pooled evaluation set over the same selected writers.
    """

    def __init__(
        self,
        *,
        split: SplitName,
        dataset_name: str = "flwrlabs/femnist",
        cache_dir: Path | str | None = Path("experiments/femnist/data/cache"),
        selected_clients_path: Path | str | None = None,
        n_clients: int = 100,
        train_fraction: float = 0.8,
        seed: int = 20260524,
        min_train_samples: int = 100,
        min_test_samples: int = 20,
        image_layout: ImageLayout = "cnn",
        max_samples_per_client: int | None = None,
        local_files_only: bool = False,
    ) -> None:
        _validate_init_args(split, n_clients, image_layout, max_samples_per_client)

        self.split = split
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.selected_clients_path = Path(selected_clients_path) if selected_clients_path is not None else None
        self.requested_n_clients = n_clients
        self.train_fraction = train_fraction
        self.seed = seed
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        self.image_layout = image_layout
        self.max_samples_per_client = max_samples_per_client
        self.local_files_only = local_files_only

        self._partitions: list[Dataset] | None = None

    @property
    def n_samples(self) -> int:
        """Return the number of datapoints in the selected split."""
        return sum(len(partition) for partition in self.get_partitions())

    @property
    def n_partitions(self) -> int:
        """Return the number of selected FEMNIST writers."""
        return len(self.selected_writer_ids)

    @property
    def n_features(self) -> int:
        """Return the flattened image feature count."""
        return 28 * 28

    @property
    def n_targets(self) -> int:
        """Return the number of FEMNIST classes."""
        return 62

    @cached_property
    def selected_writer_ids(self) -> list[str]:
        """Return the deterministic selected writer IDs."""
        return _load_or_select_writer_ids(
            metadata=self.metadata,
            selected_clients_path=self.selected_clients_path,
            n_clients=self.requested_n_clients,
            min_train_samples=self.min_train_samples,
            min_test_samples=self.min_test_samples,
            seed=self.seed,
        )

    @cached_property
    def metadata(self) -> pd.DataFrame:
        """Return FEMNIST metadata with deterministic train/test labels."""
        metadata = load_huggingface_metadata(self.cache_dir, local_files_only=self.local_files_only)
        return add_seeded_per_writer_train_test_split(
            metadata,
            train_fraction=self.train_fraction,
            seed=self.seed,
        )

    @cached_property
    def hf_dataset(self) -> _HuggingFaceDataset:
        """Return the underlying Hugging Face dataset."""
        with huggingface_offline_mode(self.local_files_only):
            try:
                from datasets import (  # type: ignore[import-untyped]  # noqa: PLC0415
                    DownloadConfig,
                    load_dataset,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "FEMNISTDatasetHandler requires the optional 'datasets' package. "
                    "Install it with: .venv\\Scripts\\python.exe -m pip install datasets"
                ) from exc

            return cast(
                "_HuggingFaceDataset",
                load_dataset(
                    self.dataset_name,
                    split="train",
                    cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                    download_config=DownloadConfig(local_files_only=self.local_files_only),
                ),
            )

    def get_datapoints(self) -> Dataset:
        """Return the selected split as one pooled dataset."""
        return [datapoint for partition in self.get_partitions() for datapoint in partition]

    def get_partitions(self) -> Sequence[Dataset]:
        """Return one selected FEMNIST writer dataset per partition."""
        if self._partitions is None:
            self._partitions = [self._build_writer_partition(writer_id) for writer_id in self.selected_writer_ids]
        return self._partitions

    def _build_writer_partition(self, writer_id: str) -> Dataset:
        writer_rows = self._split_rows_for_writer(writer_id)
        row_indices = writer_rows["row_index"].to_list()
        return [self._row_to_datapoint(int(row_index)) for row_index in row_indices]

    def _split_rows_for_writer(self, writer_id: str) -> pd.DataFrame:
        writer_rows = self.metadata[(self.metadata["writer_id"] == writer_id) & (self.metadata["split"] == self.split)]
        writer_rows = writer_rows.sort_values("row_index")

        if self.max_samples_per_client is not None:
            writer_rows = writer_rows.head(self.max_samples_per_client)

        return writer_rows

    def _row_to_datapoint(self, row_index: int) -> Datapoint:
        import torch  # noqa: PLC0415

        row = self.hf_dataset[row_index]
        image = _image_to_tensor(row["image"], layout=self.image_layout)
        character = cast("int | str", row["character"])
        label = torch.tensor(int(character), dtype=torch.long)
        return image, label


def _validate_init_args(
    split: SplitName,
    n_clients: int,
    image_layout: ImageLayout,
    max_samples_per_client: int | None,
) -> None:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if image_layout not in ("cnn", "flat"):
        raise ValueError(f"image_layout must be 'cnn' or 'flat', got {image_layout!r}")
    if n_clients <= 0:
        raise ValueError(f"n_clients must be positive, got {n_clients}")
    if max_samples_per_client is not None and max_samples_per_client <= 0:
        raise ValueError(f"max_samples_per_client must be positive, got {max_samples_per_client}")


def _load_or_select_writer_ids(
    *,
    metadata: pd.DataFrame,
    selected_clients_path: Path | None,
    n_clients: int,
    min_train_samples: int,
    min_test_samples: int,
    seed: int,
) -> list[str]:
    if selected_clients_path is not None and selected_clients_path.exists():
        selected = pd.read_csv(selected_clients_path)
        if "writer_id" not in selected.columns:
            raise ValueError(f"selected clients file must contain a writer_id column: {selected_clients_path}")
        writer_ids = selected["writer_id"].astype(str).tolist()
        if len(writer_ids) < n_clients:
            raise ValueError(
                f"selected clients file contains {len(writer_ids)} clients, but n_clients={n_clients} was requested."
            )
        return writer_ids[:n_clients]

    stats = client_stats(metadata)
    selected = choose_candidate_clients(
        stats,
        candidate_clients=n_clients,
        min_train_samples=min_train_samples,
        min_test_samples=min_test_samples,
        seed=seed,
    )
    return selected["writer_id"].astype(str).tolist()


def _image_to_tensor(image: object, *, layout: ImageLayout) -> torch.Tensor:
    import torch  # noqa: PLC0415

    if hasattr(image, "convert"):
        converted_image = cast("_ConvertibleImage", image).convert("L")
        array = np.asarray(converted_image, dtype=np.float32)
    else:
        array = np.asarray(image, dtype=np.float32)

    if array.ndim == 3:
        array = array[..., 0]

    array = array / 255.0
    tensor = torch.from_numpy(array).to(dtype=torch.float32)
    if layout == "flat":
        return tensor.reshape(-1)
    if layout == "cnn":
        return tensor.unsqueeze(0)
    raise ValueError(f"image_layout must be 'cnn' or 'flat', got {layout!r}")
