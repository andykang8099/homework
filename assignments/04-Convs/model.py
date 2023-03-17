import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """_summary_"""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=20,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(20)
        # self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # (27-2)/2 + 1 = 13
        # self.fc1 = nn.Linear(12 * 16 * 16, 256)
        self.fc1 = nn.Linear(20 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_"""

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # x = F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 20 * 8 * 8)
        x = self.fc1(x)
        return x
