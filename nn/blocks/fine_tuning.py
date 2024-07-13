"""Finetuning blocks."""


import gin
import torch
from torch import nn

from . import mlp


@gin.register(module="blocks")
class SSFAda(nn.Module):
    """Scale and shift factor for adaptive normalization Finetuning block.
    As described in https://arxiv.org/pdf/2210.08823"""

    def __init__(self, dim: int, *args, **kwargs) -> None:
        """Constructor.

        Args:
            dim (int): Dimension of the input tensor.
        """
        super().__init__(*args, **kwargs)
        self.ssf_scale = nn.Parameter(torch.ones(dim))
        self.ssf_shift = nn.Parameter(torch.zeros(dim))
        self.init_parameters()

    def init_parameters(self):
        """Initialize the parameters."""
        nn.init.normal_(self.ssf_scale, mean=1, std=.02)
        nn.init.normal_(self.ssf_shift, std=.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        # TODO: change those conditions on the shape of the input tensor
        if x.shape[-1] == self.ssf_scale.shape[0]:
            return x * self.ssf_scale + self.ssf_shift
        if x.shape[1] == self.ssf_scale.shape[0]:
            return (x * self.ssf_scale.view(1, -1, 1, 1) +
                    self.ssf_shift.view(1, -1, 1, 1))
        raise ValueError('the input tensor shape does not match the '
                         'shape of the scale factor.')


@gin.register(module="blocks")
class FiLM(nn.Module):
    """Feature-wise Linear Modulation blocks. This can be used to condition
    CNNs on some external features.
    As described in https://arxiv.org/pdf/1709.07871"""

    def __init__(
        self,
        input_features: int,
        out_features: int,
        *args,
        hidden_features: int | list[int] | None = [64, 64, 64, 64],
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            input_features (int): Number of input features.
            out_features (int): Number of output features.
            hidden_features (int | list[int] | None): Number of hidden features.
        """
        super().__init__(*args, **kwargs)
        self.film_scale_mlp = mlp.MLP(
            in_features=input_features,
            out_features=out_features,
            hidden_features=hidden_features,
            activation=nn.ReLU,
            normalization=nn.LayerNorm,
        )
        self.film_shift_mlp = mlp.MLP(
            in_features=input_features,
            out_features=out_features,
            hidden_features=hidden_features,
            activation=nn.ReLU,
            normalization=nn.LayerNorm,
        )

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        scale = self.film_scale_mlp(features)
        shift = self.film_shift_mlp(features)
        if x.shape[-1] == scale.shape[1]:
            return x * scale[:, None, None, :] + shift[:, None, None, :]
        if x.shape[1] == scale.shape[1]:
            return (x * scale[:, :, None, None] + shift[:, :, None, None])
        raise ValueError('the input tensor shape does not match the '
                         'shape of the scale factor.')
