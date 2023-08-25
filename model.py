import torch
import torch.nn as nn


class FashionMNISTModelV0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convstack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten(start_dim=1),
        )

        self.linstack = nn.Sequential(
            nn.Linear(in_features=256, out_features=10),
            nn.Softmax(dim=1),
        )

    def forward(self, X):
        y = self.convstack(X)
        y = self.linstack(y)
        return y
