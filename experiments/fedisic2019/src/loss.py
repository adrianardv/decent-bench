from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.nn import functional as F


FLAMBY_FEDISIC2019_ALPHA = torch.tensor(
    [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443],
    dtype=torch.float32,
)


class WeightedFocalLoss(torch.nn.Module):
    """Weighted focal loss used for Fed-ISIC2019."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("alpha", (FLAMBY_FEDISIC2019_ALPHA if alpha is None else alpha).to(torch.float32))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return weighted focal loss for class logits and integer targets."""
        targets = targets.view(-1).long()
        log_prob = F.log_softmax(logits, dim=1)
        target_log_prob = log_prob.gather(1, targets.view(-1, 1)).view(-1)
        target_prob = target_log_prob.exp()
        alpha = self.alpha.to(logits.device).gather(0, targets)
        loss = -alpha * ((1.0 - target_prob) ** self.gamma) * target_log_prob
        return loss.mean()


def class_weights_from_labels(labels: Iterable[int], num_classes: int) -> torch.Tensor:
    """Compute FLamby benchmark-style N / n_c class weights from integer labels."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        counts[int(label)] += 1
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).view(-1).tolist()
        raise ValueError(f"Cannot compute class weights because classes {missing} have zero samples.")
    return counts.sum() / counts
