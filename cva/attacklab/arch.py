"""Architectures the loaders can resolve. Two, deliberately different, so a reference
model is never a sibling of the artifact under test."""
from __future__ import annotations

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 5, width: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(width * 2, width * 4, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        return self.fc(self.flat(x))


class TinyMLP(nn.Module):
    """Structurally unrelated to SmallCNN — used for decorrelated references."""

    def __init__(self, num_classes: int = 5, hidden: int = 128):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden // 2, num_classes)

    def forward(self, x):
        x = self.act(self.fc1(self.flat(x)))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class WideCNN(nn.Module):
    """Structurally decorrelated from SmallCNN — 5x5 kernels, two blocks, average
    pooling, different head — while still able to learn the task. Decorrelation must not
    be bought by using an architecture that cannot solve the problem."""

    def __init__(self, num_classes: int = 8, width: int = 24):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(width, width * 2, 5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(width * 2 * 4, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.fc2(self.act(self.fc1(self.flat(x))))


ARCH_REGISTRY = {"SmallCNN": SmallCNN, "TinyMLP": TinyMLP, "WideCNN": WideCNN}
