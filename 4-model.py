"""CNN architecture for chest X-ray pneumonia classification.

Designed for small grayscale medical images (28x28, PneumoniaMNIST).
"""

import torch
import torch.nn as nn


class PneumoniaCNN(nn.Module):
    """Compact CNN for small grayscale medical images.

    Three conv blocks with batch normalization followed by global average
    pooling keep the parameter count around 90k, which suits the small
    PneumoniaMNIST training set (4,708 images) and limits overfitting.
    Global average pooling instead of a large fully-connected head makes the
    model roughly input-size agnostic and cuts parameters sharply.
    """

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)
