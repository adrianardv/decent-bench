from __future__ import annotations

import torch


class FEMNISTCNN(torch.nn.Module):
    """Convolutional classifier used by the thesis FEMNIST experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 62),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for FEMNIST class labels."""
        return self.classifier(self.features(x))
