""" Loss wrapper for deep framework. """

import torch
from torch import nn


class LossWrapper(nn.Module):
    """LossWrapper.

    Args:
        loss_fn (callable): loss function.
        **kwargs: additional arguments for the loss function.
    """

    def __init__(
            self,
            loss_fn: nn.Module,
            channels: list[int] | None = None,
            activation: nn.Module | None = None,
            **kwargs
    ):
        """Constructor.

        Args:
            loss_fn (nn.Module): loss function.
            channels (list[int]): channels to consider. Default to None.
            activation_fn (nn.Module): activation function. Default to None.
            **kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        self.channels = channels
        self.loss_fn = loss_fn
        self.activation = activation

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        *args,
        ** kwargs
    ) -> torch.Tensor:
        """Computation of the loss.

        Args:
            input (torch.Tensor): Network predictions.
            target (torch.Tensor): Target Tensor.

        Returns:
            torch.Tensor: computed loss.
        """
        if self.channels is not None:
            inputs = inputs[:, self.channels]
            target = target[:, self.channels]
        if self.activation is not None:
            inputs = self.activation(inputs)
        return self.loss_fn(inputs, target)
