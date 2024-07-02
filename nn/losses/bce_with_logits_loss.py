""" BCEWithLogitsLoss class. """

from typing import Any

import gin
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import loss


@gin.register(module="losses")
class BCEWithLogitsLoss(loss._Loss):
    """BCEWithLogitsLoss.

    Args:
        **kwargs: additional arguments for the loss function.
    """

    def __init__(
            self,
            weight: torch.Tensor | None = None,
            size_average: Any | None = None,
            reduce: Any | None = None,
            reduction: str = 'mean',
            pos_weight: torch.Tensor | None = None,
            channels: list[int] | None = None,
            activation: nn.Module | None = None,
            **kwargs):
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
            pos_weight (torch.Tensor): a weight of positive examples. Must be a vector with length equal to
                the number of classes. Default to None.
            channels (list[int]): channels to consider. Default to None.
            **kwargs: additional arguments.    
        """

        super().__init__(size_average, reduce, reduction)
        self.register_buffer('pos_weight', pos_weight)
        self.pos_weight: torch.Tensor | None
        self.channels = channels
        self.activation = activation
        self.pos_weight = pos_weight

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        *args,
        weight: torch.Tensor | None = None,
        ** kwargs
    ) -> torch.Tensor:
        """Computation of the loss.

        Args:
            input (torch.Tensor): Network predictions.
            target (torch.Tensor): Target Tensor.
            weight (torch.Tensor): a manual rescaling weight given to the loss 
                of each batch element.

        Returns:
            torch.Tensor: computed loss.
        """
        if self.channels is not None:
            inputs = inputs[:, self.channels]
            target = target[:, self.channels]
        if self.activation is not None:
            inputs = self.activation(inputs)
        return F.binary_cross_entropy_with_logits(
            inputs,
            target,
            weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )
