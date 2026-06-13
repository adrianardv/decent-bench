from __future__ import annotations

import random
from typing import Any

import numpy as np


IMAGE_SIZE = 200


def build_train_transform(image_size: int = IMAGE_SIZE) -> Any:
    """Return the FLamby-style Fed-ISIC2019 training transform."""
    albumentations = _import_albumentations()
    return albumentations.Compose(
        [
            albumentations.RandomScale(0.07),
            albumentations.Rotate(50),
            albumentations.RandomBrightnessContrast(0.15, 0.1),
            _flip(albumentations),
            albumentations.Affine(shear=0.1),
            albumentations.RandomCrop(height=image_size, width=image_size),
            _coarse_dropout(albumentations, max_holes=random.randint(1, 8), hole_size=16),
            _normalize(albumentations),
        ]
    )


def build_test_transform(image_size: int = IMAGE_SIZE) -> Any:
    """Return the deterministic FLamby-style Fed-ISIC2019 evaluation transform."""
    albumentations = _import_albumentations()
    return albumentations.Compose(
        [
            albumentations.CenterCrop(height=image_size, width=image_size),
            _normalize(albumentations),
        ]
    )


def image_to_chw_float32(image: object, transform: Any) -> np.ndarray:
    """Apply *transform* and return a C x H x W float32 array."""
    image_array = _image_to_rgb_array(image)
    augmented = transform(image=image_array)
    image_array = augmented["image"]
    return np.transpose(image_array, (2, 0, 1)).astype(np.float32)


def _image_to_rgb_array(image: object) -> np.ndarray:
    if hasattr(image, "convert"):
        image = image.convert("RGB")  # type: ignore[attr-defined]
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return array.astype(np.uint8, copy=False)


def _import_albumentations() -> Any:
    try:
        import albumentations  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Fed-ISIC2019 preprocessing requires albumentations. "
            "Install experiment dependencies with: .venv\\Scripts\\python.exe -m pip install -e .[dev]"
        ) from exc
    return albumentations


def _normalize(albumentations: Any) -> Any:
    return albumentations.Normalize(p=1.0)


def _flip(albumentations: Any) -> Any:
    if hasattr(albumentations, "Flip"):
        return albumentations.Flip(p=0.5)
    return albumentations.OneOf(
        [
            albumentations.HorizontalFlip(p=1.0),
            albumentations.VerticalFlip(p=1.0),
        ],
        p=0.5,
    )


def _coarse_dropout(albumentations: Any, *, max_holes: int, hole_size: int) -> Any:
    """Construct CoarseDropout across albumentations 1.x and 2.x APIs."""
    try:
        return albumentations.CoarseDropout(
            num_holes_range=(1, max_holes),
            hole_height_range=(hole_size, hole_size),
            hole_width_range=(hole_size, hole_size),
            p=1.0,
        )
    except TypeError:
        try:
            return albumentations.CoarseDropout(
                max_holes=max_holes,
                max_height=hole_size,
                max_width=hole_size,
                p=1.0,
            )
        except TypeError:
            return albumentations.CoarseDropout(max_holes, hole_size, hole_size)
