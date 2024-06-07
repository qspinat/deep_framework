"""UNet implementation."""

from typing import Sequence

import gin
import torch
from torch import nn
from torch.nn.modules import pooling

from ..blocks import blocks
from .. import utils


@gin.register(module="models")
class UNet(nn.Module):
    """UNet model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_filters: int = 32,
        kernel_size: int | Sequence[int] = 3,
        depth: int = 2,
        conv_per_level: int = 2,
        stride: int | Sequence[int] = 2,
        normalization: type[nn.Module] = nn.BatchNorm2d,
        linear_upsampling: bool = False,
        transformer_block: type[nn.Module] | None = None,
        activation: type[nn.Module] = nn.ReLU,
        out_activation: type[nn.Module] = nn.Identity,
        dropout: type[nn.modules.dropout._DropoutNd] | None = None,
        dim: int = 2,
        *args,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            base_filters (int): Number of base filters. Default to 32.
            kernel_size (int | Sequence[int]): Kernel size. Default to 3.
            depth (int): Depth of the UNet. Default to 2.
            conv_per_level (int): Number of convolutional layers per level. Default to 2.
            stride (int | Sequence[int]): Stride. Default to 2.
            normalization (torch.nn.Module): Normalization layer. Default to BatchNorm2d.
            linear_upsampling (bool): Whether to use linear upsampling or not. Default to False.
            transformer_block (torch.nn.Module | None): Transformer block. Default to None.
            activation (torch.nn.Module): Activation layer. Default to ReLU.
            out_activation (torch.nn.Module): Output activation layer. Default to Identity.
            dropout (torch.nn.modules.dropout._DropoutNd | None): Dropout layer. Default to None.
            dim (int): Dimension of the input. Default to 2.
        """
        super().__init__(*args, **kwargs)

        if type(kernel_size) == int:
            kernel_size = [kernel_size for i in range(dim)]
        if type(stride) == int:
            stride = [stride for i in range(dim)]

        act = activation() if activation is not None else None

        self.input_conv = nn.Sequential(
            blocks.ConvUnit(
                conv=utils.get_conv_nd(dim)(
                    in_channels=in_channels,
                    out_channels=base_filters,
                    kernel_size=kernel_size,
                    padding=[(k-1)//2 for k in kernel_size]
                ),
                normalization=normalization(base_filters),
                activation=act
            ),
            DownBlock(
                in_channels=base_filters,
                out_channels=base_filters,
                kernel_size=kernel_size,
                stride=[1 for j in range(dim)],
                n_convs=conv_per_level,
                normalization=normalization,
                transformer_block=None,
                activation=activation,
                dropout=dropout,
                dim=dim,
            ))

        self.output_conv = utils.get_conv_nd(dim)(
            in_channels=base_filters,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.out_act = out_activation()

        self.down_path = []
        for i in range(depth):
            self.down_path.append(
                DownBlock(
                    in_channels=base_filters*2**i,
                    out_channels=base_filters*2**(i+1),
                    kernel_size=kernel_size,
                    stride=stride,
                    n_convs=conv_per_level,
                    normalization=normalization,
                    transformer_block=(
                        transformer_block if i == (depth-1) else None),
                    activation=activation,
                    dropout=dropout,
                    dim=dim,
                )
            )
        self.down_path = nn.ModuleList(self.down_path)

        self.up_path = []
        for i in range(depth):
            self.up_path.append(
                UpBlock(
                    in_channels=base_filters*2**(i+1),
                    out_channels=base_filters*2**i,
                    kernel_size=kernel_size,
                    upsample=True,
                    upsample_factors=tuple(stride),
                    linear_upsampling=linear_upsampling,
                    n_convs=conv_per_level,
                    normalization=normalization,
                    activation=activation,
                    dropout=dropout,
                    dim=dim,
                )
            )
        self.up_path.reverse()
        self.up_path = nn.ModuleList(self.up_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        x = self.input_conv(x)
        skip_connections.append(x.clone())
        for block in self.down_path:
            x = block(x)
            skip_connections.append(x.clone())
        skip_connections.pop()
        for block in self.up_path:
            x = block(x, skip_connections.pop())
        x = self.output_conv(x)
        return self.out_act(x)


class DownBlock(nn.Module):
    """UNet down block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        n_convs: int,
        normalization: type[nn.Module],
        transformer_block: type[nn.Module],  # TODO
        dropout: type[nn.modules.dropout._DropoutNd] | None,
        activation: type[nn.Module],
        dim: int,
        *args,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Sequence[int]): Kernel size.
            stride (Sequence[int]): Stride.
            n_convs (int): Number of convolutional layers.
            normalization (torch.nn.Module): Normalization layer.
            transformer_block (torch.nn.Module): Transformer block.
            dropout (torch.nn.modules.dropout._DropoutNd): Dropout layer.
            activation (torch.nn.Module): Activation layer.
            dim (int): Dimension of the input.
        """
        super().__init__(*args, **kwargs)
        downsample = any([s != 1 for s in stride])
        self.downsample = (
            nn.Identity() if not downsample
            else blocks.ConvUnit(
                conv=utils.get_conv_nd(dim)(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=stride,
                    stride=stride,
                    padding=0,
                ),
                normalization=normalization(out_channels),
                activation=activation() if activation is not None else None,
                dropout=dropout() if dropout is not None else None,
            )
        )
        self.res_blocks = nn.Sequential(*[
            blocks.ConvUnit(
                conv=utils.get_conv_nd(dim)(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=[
                        (k-1)//2 for k in kernel_size],
                    stride=1),
                normalization=normalization(out_channels),
                activation=activation() if activation is not None else None,
                dropout=dropout() if dropout is not None else None,
            ) for i in range(n_convs)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.res_blocks(x)
        return x


class UpBlock(nn.Module):
    """UNet up block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        upsample_factors: tuple[int],
        linear_upsampling: bool,
        n_convs: int,
        normalization: type[nn.Module],
        activation: type[nn.Module],
        dropout: type[nn.modules.dropout._DropoutNd] | None,
        dim: int,
        *args,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Sequence[int]): Kernel size.
            upsample_factors (tuple[int]): Upsample factors.
            linear_upsampling (bool): Whether to use linear upsampling or not.
            n_convs (int): Number of convolutional layers.
            normalization (torch.nn.Module): Normalization layer.
            activation (torch.nn.Module): Activation layer.
            dropout (torch.nn.modules.dropout._DropoutNd): Dropout layer.
            dim (int): Dimension of the input.
        """
        super().__init__(*args, **kwargs)
        if linear_upsampling:
            self.upsample = nn.Sequential(
                nn.Upsample(
                    scale_factor=upsample_factors,
                    mode=("linear" if dim == 1
                          else "bilinear" if dim == 2
                          else "trilinear")
                ),
                blocks.ConvUnit(
                    conv=utils.get_conv_nd(dim)(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=[(k-1)//2 for k in kernel_size],
                    ),
                    normalization=normalization(out_channels),
                    activation=activation() if activation is not None else None,
                    dropout=dropout() if dropout is not None else None,
                )
            )
        else:
            self.upsample = blocks.ConvUnit(
                conv=utils.get_conv_nd(dim, transposed=True)(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=upsample_factors,
                    stride=upsample_factors,
                ),
                normalization=normalization(out_channels),
                activation=activation() if activation is not None else None,
                dropout=dropout() if dropout is not None else None,
            )

        self.res_blocks = nn.Sequential(*[
            blocks.ConvUnit(
                conv=utils.get_conv_nd(dim)(
                    in_channels=2*out_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=[
                        (k-1)//2 for k in kernel_size],
                    stride=1),
                normalization=normalization(out_channels),
                activation=activation() if activation is not None else None,
                dropout=dropout() if dropout is not None else None,
            ) for i in range(n_convs)
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat((skip, x), dim=1)
        x = self.res_blocks(x)
        return x
