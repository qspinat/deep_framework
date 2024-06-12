""" Rescaled Sigmoid Activation Function. """

import gin
import torch
from torch import nn


@gin.register(module="activation")
class RescaledSigmoid(nn.Module):
    """ Rescaled Sigmoid Activation Function. """

    def __init__(self, scale: float, shift: float) -> None:
        super().__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale*torch.sigmoid(x) + self.shift
