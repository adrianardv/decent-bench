from __future__ import annotations

import torch


class FedISIC2019EfficientNet(torch.nn.Module):
    """EfficientNet-B0 classifier for Fed-ISIC2019."""

    def __init__(self, num_classes: int = 8, *, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "FedISIC2019EfficientNet requires torchvision. "
                "Install experiment dependencies with: .venv\\Scripts\\python.exe -m pip install -e .[dev]"
            ) from exc

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.base_model = efficientnet_b0(weights=weights)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for Fed-ISIC2019 class labels."""
        return self.base_model(x)


class FedISICSmallCNN(torch.nn.Module):
    """Small CNN for smoke/debug runs; not the main Fed-ISIC2019 model."""

    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for Fed-ISIC2019 class labels."""
        features = self.features(x)
        return self.classifier(torch.flatten(features, 1))


def build_model(name: str, *, num_classes: int = 8, pretrained: bool = True) -> torch.nn.Module:
    """Build a Fed-ISIC2019 model by name."""
    if name == "efficientnet_b0":
        return FedISIC2019EfficientNet(num_classes=num_classes, pretrained=pretrained)
    if name == "small_cnn":
        return FedISICSmallCNN(num_classes=num_classes)
    raise ValueError(f"Unsupported Fed-ISIC2019 model name: {name!r}")
