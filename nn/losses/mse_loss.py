""" L1 loss. """

from typing import Any

import gin
from torch import nn

from . import loss_wrapper


@gin.register(module="losses")
class MSELoss(loss_wrapper.LossWrapper):
    """MSE loss."""

    def __init__(
        self,
        size_average: Any | None = None,
        reduce: Any | None = None,
        reduction: str = 'mean',
        channels: list[int] | None = None,
        activation: nn.Module | None = None,
        **kwargs
    ):
        """Constructor.

        Args:
            size_average (Any): deprecated (see reduction). By default, the losses are averaged over
                each loss element in the batch. Note that for some losses, there multiple elements per
                sample. If the field size_average is set to False, the losses are instead summed for
                each minibatch. Ignored when reduce is False. Default to None.
            reduce (Any): deprecated (see reduction). By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce is False, returns
                a loss per batch element instead and ignores size_average. Default to None.
            reduction (str): specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the
                output will be divided by the number of elements in the output, 'sum': the output will be
                summed. Default to 'mean'.
            channels (list[int]): channels to consider. Default to None.
            activation (nn.Module): activation function. Default to None.
            **kwargs: additional arguments.
        """
        mse = nn.MSELoss(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction
        )
        super().__init__(loss_fn=mse, channels=channels, activation=activation, **kwargs)
