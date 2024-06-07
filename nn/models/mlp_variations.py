"""Variations of MLP models."""

import torch

import gin

from ..blocks import mlp


@gin.register(module="models")
class MaxMLP(mlp.MLP):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.amax(dim=0, keepdim=True)
        return super().forward(input)


@gin.register(module="models")
class AvgMLP(mlp.MLP):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.mean(dim=0, keepdim=True)
        return super().forward(input)
