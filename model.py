import torch
import torch.nn as nn


class FashionMNISTModelV0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convstack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )
        self.linstack = nn.Sequential(
            nn.Linear(in_features=10000, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=1),
        )

    def forward(self, X):
        y = self.convstack(X)
        y = self.linstack(y)
        return y
