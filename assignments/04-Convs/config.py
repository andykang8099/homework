from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize


class CONFIG:
    batch_size = 200
    num_epochs = 6

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=3e-3)

    transforms = Compose(
        [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])]
    )
