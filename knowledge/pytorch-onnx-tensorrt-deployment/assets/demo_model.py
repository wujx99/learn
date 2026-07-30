"""A small deterministic CNN shared by the deployment exercises."""

from __future__ import annotations

import torch
from torch import nn


class DemoNet(nn.Module):
    """Input: NCHW float tensor [N, 3, 32, 32]; output: [N, 10]."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(torch.flatten(features, 1))


def make_model(seed: int = 42) -> DemoNet:
    """Construct reproducible weights without depending on a training job."""
    torch.manual_seed(seed)
    return DemoNet()
