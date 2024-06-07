""" Multi Layer Perceptron (MLP) implementation. """

from typing import Sequence

import gin
from torch import nn


class Perceptron(nn.Sequential):
    """Perceptron. Composed of a linear layer, a normalization layer, an
    activation layer and a dropout layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        normalization: nn.Module | None = None,
        activation: nn.Module | None = None,
        dropout: nn.modules.dropout._DropoutNd | None = None,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Whether to use bias or not. Default to True.
            normalization (torch.nn.Module | None): Normalization layer. Default to None.
            activation (torch.nn.Module | None): Activation layer. Default to None.
            dropout (torch.nn.modules.dropout._DropoutNd | None): Dropout layer. Default to None.
        """
        linear = nn.Linear(in_features=in_features,
                           out_features=out_features,
                           bias=bias)
        sequence = [linear]
        if normalization is not None:
            sequence.append(normalization)
        if activation is not None:
            sequence.append(activation)
        if dropout is not None:
            sequence.append(dropout)
        super().__init__(*sequence, **kwargs)


@gin.register(module="blocks")
class MLP(nn.Sequential):
    """Multi Layer Perceptron."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | Sequence[int],
        bias: bool = True,
        normalization: type[nn.Module] | None = None,
        activation: type[nn.Module] | None = None,
        dropout: nn.modules.dropout._DropoutNd | None = None,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            hidden_features (int | Sequence[int]): Number of hidden features.
            bias (bool): Whether to use bias or not. Default to True.
            normalization (torch.nn.Module | None): Normalization layer. Default to None.
            activation (torch.nn.Module | None): Activation layer. Default to None.
            dropout (torch.nn.modules.dropout._DropoutNd | None): Dropout layer. Default to None.
        """

        normalization = (
            nn.Identity if normalization is None else normalization)
        hidden_features = (
            hidden_features if hidden_features is not None else in_features)
        features = [in_features, *hidden_features, out_features]

        layers = [
            Perceptron(
                in_features=features[i],
                out_features=features[i+1],
                bias=bias,
                # no normalization, activation nor dropout on last layer
                normalization=(normalization(features[i+1])
                               if i < (len(features)-2) else None),
                activation=activation() if (i < len(features)-2) else None,
                dropout=dropout if i < (len(features)-2) else None,
            ) for i in range(len(features)-1)
        ]

        super().__init__(*layers, **kwargs)
