from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.fedisic2019.src import FedISICDatasetHandler


pytest.importorskip("albumentations")


class _FakeLabelFeature:
    names = ["class_0", "class_1"]


class _FakeDataset:
    column_names = ["image", "center", "label"]
    features = {"image": object(), "label": _FakeLabelFeature()}

    def __init__(self) -> None:
        self.rows = [
            {"image": np.full((224, 224, 3), 128, dtype=np.uint8), "center": 0, "label": 0},
            {"image": np.full((224, 224, 3), 64, dtype=np.uint8), "center": 1, "label": 1},
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]

    def select_columns(self, columns: list[str]) -> object:
        return SimpleNamespace(
            to_pandas=lambda: pd.DataFrame([{column: row[column] for column in columns} for row in self.rows])
        )


def test_fedisic_handler_partitions_by_center_and_returns_chw_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dataset = _FakeDataset()
    handler = FedISICDatasetHandler(split="train")
    monkeypatch.setattr(handler, "_load_hf_dataset", lambda decode_images: fake_dataset)

    partitions = handler.get_partitions()
    image, label = partitions[0][0]

    assert handler.center_ids == [0, 1]
    assert handler.class_names == ["class_0", "class_1"]
    assert len(partitions) == 2
    assert [len(partition) for partition in partitions] == [1, 1]
    assert tuple(image.shape) == (3, 200, 200)
    assert image.dtype.is_floating_point
    assert label.dtype.is_floating_point is False
    assert int(label) == 0
