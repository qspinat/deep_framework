"""Basic blocks implementations."""

from typing import Sequence

import gin
import torch
from torch import nn
from torch.nn.modules import batchnorm
from torch.nn.modules import conv
from torch.nn.modules import dropout

from .. import utils


class ConvUnit(nn.Sequential):
    """Convolutional unit. Composed of a convolutional layer, a 
    normalization layer, an activation layer and a dropout layer."""

    def __init__(
        self,
        conv: conv._ConvNd,
        normalization: batchnorm._NormBase | None = None,
        activation: nn.Module | None = nn.ReLU(),
        dropout: dropout._DropoutNd | None = None,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            conv (torch.nn.modules.conv._ConvNd): Convolutional layer.
            normalization (torch.nn.modules.batchnorm._NormBase | None): 
                Normalization layer. Default to None.
            activation (torch.nn.Module | None): Activation layer. Default to ReLU.
            dropout (torch.nn.modules.dropout._DropoutNd | None): Dropout layer. 
                Default to None."""
        sequence = [conv]
        if normalization is not None:
            sequence.append(normalization)
        if activation is not None:
            sequence.append(activation)
        if dropout is not None:
            sequence.append(dropout)
        super().__init__(*sequence, **kwargs)


@gin.register(module="nn")
class ResNetBlock(nn.Module):
    """Basic ResNet block. Composed of two convolutional units and a projection"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        strided_kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 1,
        normalization: type[batchnorm._NormBase] | None = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
        out_activation: int | Sequence[int] | None = nn.ReLU,
        dropout: type[dropout._DropoutNd] | None = None,
        dim: int = 2,
        *args,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int | Sequence[int]): Kernel size. Default to 3.
            strided_kernel_size (int | Sequence[int]): Strided kernel size. Default to 2.
            stride (int | Sequence[int]): Stride. Default to 1.
            normalization (torch.nn.modules.batchnorm._NormBase | None): Normalization layer. 
                Default to nn.BatchNorm2d.
            activation (torch.nn.Module): Activation layer. Default to ReLU.
            out_activation (int | Sequence[int] | None): Output activation layer. Default to ReLU.
            dropout (torch.nn.modules.dropout._DropoutNd | None): Dropout layer. Default to None.
            dim (int): Dimension of the convolution. Default to 2.
        """
        super().__init__(*args, **kwargs)
        if type(kernel_size) == int:
            kernel_size = [kernel_size for i in range(dim)]
        if type(strided_kernel_size) == int:
            strided_kernel_size = [strided_kernel_size for i in range(dim)]
        if type(stride) == int:
            stride = [stride for i in range(dim)]

        downsample = any([s != 1 for s in stride])

        self.conv1 = ConvUnit(
            conv=utils.get_conv_nd(dim)(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=[k//2 for k in kernel_size],
                kernel_size=kernel_size,
                stride=stride,
            ),
            normalization=normalization(out_channels),
            activation=activation(),
            dropout=(dropout() if dropout is not None else None),
        )

        self.conv2 = ConvUnit(
            conv=utils.get_conv_nd(dim)(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=[k//2 for k in kernel_size],
            ),
            normalization=normalization(out_channels),
            activation=None,
            dropout=(dropout() if dropout is not None else None),
        )

        self.projection = ConvUnit(
            conv=utils.get_conv_nd(dim)(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=strided_kernel_size,
                padding=[k//2 for k in strided_kernel_size],
                stride=stride,
            ),
            normalization=normalization(out_channels),
            activation=None,
            dropout=(dropout() if dropout is not None else None),
        ) if downsample else nn.Identity()

        self.act = (out_activation() if out_activation
                    is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x.clone())
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        if self.act is not None:
            x = self.act(x)
        return x


@gin.register(module="nn")
class BottleNeckResNetBlock(nn.Module):
    """BottleNeck ResNet block. Composed of three convolutional units, the first and third 
    one having a kernel size of 1, and a projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        strided_kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 1,
        normalization: type[batchnorm._NormBase] | None = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
        out_activation: int | Sequence[int] | None = nn.ReLU,
        dropout: type[dropout._DropoutNd] | None = None,
        dim: int = 2,
        *args,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int | Sequence[int]): Kernel size. Default to 3.
            strided_kernel_size (int | Sequence[int]): Strided kernel size. Default to 2.
            stride (int | Sequence[int]): Stride. Default to 1.
            normalization (torch.nn.modules.batchnorm._NormBase | None): Normalization layer. 
                Default to nn.BatchNorm2d.
            activation (torch.nn.Module): Activation layer. Default to ReLU.
            out_activation (int | Sequence[int] | None): Output activation layer. Default to ReLU.
            dropout (torch.nn.modules.dropout._DropoutNd | None): Dropout layer. Default to None.
            dim (int): Dimension of the convolution. Default to 2.
        """
        super().__init__(*args, **kwargs)
        if type(kernel_size) == int:
            kernel_size = [kernel_size for i in range(dim)]
        if type(strided_kernel_size) == int:
            strided_kernel_size = [strided_kernel_size for i in range(dim)]
        if type(stride) == int:
            stride = [stride for i in range(dim)]

        downsample = any([s != 1 for s in stride])

        self.conv1 = ConvUnit(
            conv=utils.get_conv_nd(dim)(
                in_channels=in_channels,
                out_channels=out_channels//4,
                kernel_size=1,
                padding=0,
                stride=stride,
            ),
            normalization=normalization(out_channels//4),
            activation=activation(),
            dropout=(dropout() if dropout is not None else None),
        )

        self.conv2 = ConvUnit(
            conv=utils.get_conv_nd(dim)(
                in_channels=out_channels//4,
                out_channels=out_channels//4,
                kernel_size=kernel_size,
                padding=[k//2 for k in kernel_size],
            ),
            normalization=normalization(out_channels//4),
            activation=activation(),
            dropout=(dropout() if dropout is not None else None),
        )

        self.conv3 = ConvUnit(
            conv=utils.get_conv_nd(dim)(
                in_channels=out_channels//4,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            ),
            normalization=normalization(out_channels),
            activation=None,
            dropout=(dropout() if dropout is not None else None),
        )

        self.projection = ConvUnit(
            conv=utils.get_conv_nd(dim)(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=strided_kernel_size,
                padding=[k//2 for k in strided_kernel_size],
                stride=stride,
            ),
            normalization=normalization(out_channels),
            activation=None,
            dropout=(dropout() if dropout is not None else None),
        ) if downsample else nn.Identity()

        self.act = (out_activation() if out_activation
                    is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x.clone())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + residual
        if self.act is not None:
            x = self.act(x)
        return x
