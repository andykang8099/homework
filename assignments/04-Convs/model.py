import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """_model_"""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=6,
            kernel_size=2,
            stride=2,
            padding=1,
        )
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(6)

        self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # (27-2)/2 + 1 = 13
        # self.fc1 = nn.Linear(12 * 16 * 16, 256)
        self.fc2 = nn.Linear(6 * 8 * 8, 256)
        # self.fc3 = nn.Linear(512, 256)
        # self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_forward_"""

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # x = F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 6 * 8 * 8)
        x = F.relu(self.bn2(self.fc2(x)))

        # x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc1(x)
        return x
