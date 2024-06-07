"""DiceWithLogitsLoss module."""

import gin
import torch
from torch import nn


@gin.register(module="losses")
class DiceWithLogitsLoss(nn.Module):
    """DiceWithLogitsLoss.

    Args:
        eps (float): epsilon for stability. Default to  1e-6.
        upper_eps (bool): whether to add on epsilon to the numerator of the
            dice.
        batch_dice (bool): whether to see the batch as a single volume and
            compute one dice for the whole batch (True) or one dice for each
            element of the batch and then average them (False). Default to True.
        loss (bool): Dice loss (True) or dice metric (False). Default to True.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        upper_eps: bool = False,
        batch_dice: bool = True,
        loss: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.upper_eps = upper_eps
        self.batch_dice = batch_dice
        self.loss = loss

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Computation of the DiceLoss.

        Args:
            input (torch.Tensor): Network predictions.
            target (torch.Tensor): Target Tensor.

        Returns:
            torch.Tensor: computed loss.
        """
        input = torch.sigmoid(input)
        mean_fn = (torch.mean if self.batch_dice
                   else lambda x: torch.flatten(x, 1).mean(1))
        numerator = 2 * mean_fn(input * target)
        if self.upper_eps:
            numerator += self.eps
        denominator = mean_fn(input**2 + target**2) + self.eps
        out = numerator / denominator
        if self.loss:
            out = 1 - out
        return out.mean()
