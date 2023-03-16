import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """_summary_"""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_channels, out_channels=16, kernel_size=2, padding=0
        )

        # self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # (27-2)/2 + 1 = 13
        self.fc = nn.Linear(16 * 15 * 15, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_"""

        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = x.view(-1, 16 * 15 * 15)
        x = self.fc(x)
        return x
