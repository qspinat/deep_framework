"""ResNet implementation."""

from typing import Sequence

import gin
import torch
from torch import nn
from torch.nn.modules import batchnorm
from torch.nn.modules import dropout

from ..blocks import blocks
from ..blocks import transformers
from .. import utils


@gin.register(module="models")
class ResNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = 1,
                 list_depth: Sequence[int] = [3, 4, 6, 3],
                 base_filters: int = 32,
                 c_mult: Sequence[int] = [1, 2, 4, 8],
                 kernel_size: int | Sequence[int] = 3,
                 strided_kernel_size: int | Sequence[int] = 2,
                 stride: int | Sequence[int] = 2,
                 groups: int = 1,
                 normalization: type[batchnorm._NormBase] | None = nn.BatchNorm2d,
                 activation: type[nn.Module] = nn.ReLU,
                 out_activation: type[nn.Module] | None = None,
                 dropout: type[dropout._DropoutNd] | None = None,
                 res_block_cls: (
                     type[blocks.ResNetBlock] | type[blocks.BottleNeckResNetBlock]
                 ) = blocks.BottleNeckResNetBlock,
                 class_attention_block: type[transformers.ClassAttentionBlock] | None = None,
                 rel_position_input: bool = False,
                 dim: int = 2,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_filters = base_filters

        if issubclass(res_block_cls, blocks.BottleNeckResNetBlock):
            self.base_filters *= 4

        if type(kernel_size) == int:
            kernel_size = [kernel_size for i in range(dim)]
        if type(strided_kernel_size) == int:
            strided_kernel_size = [strided_kernel_size for i in range(dim)]

        if rel_position_input:
            in_channels += dim
        self.input_conv = blocks.ConvUnit(
            utils.get_conv_nd(dim)(
                in_channels=in_channels,
                out_channels=self.base_filters,
                kernel_size=strided_kernel_size,
                padding=[k//2 for k in strided_kernel_size],
                stride=stride,
                groups=groups,
            ),
            normalization=normalization(self.base_filters),
            activation=activation(),
            dropout=(dropout() if dropout is not None else None),
        )

        self.pool = utils.get_maxpool_nd(dim)(
            kernel_size=strided_kernel_size,
            stride=stride,
            padding=[k//2 for k in strided_kernel_size],
        )

        self.res_convs = []

        for i, d in enumerate(list_depth):
            self.res_convs.extend([res_block_cls(
                in_channels=self.base_filters*c_mult[i],
                out_channels=self.base_filters*c_mult[i],
                kernel_size=kernel_size,
                strided_kernel_size=strided_kernel_size,
                stride=1,
                groups=groups,
                normalization=normalization,
                activation=activation,
                out_activation=activation,
                dropout=dropout,
                dim=dim
            ) for j in range(d)])
            if i < len(list_depth)-1:
                self.res_convs.append(res_block_cls(
                    in_channels=self.base_filters*c_mult[i],
                    out_channels=self.base_filters*c_mult[i+1],
                    kernel_size=kernel_size,
                    strided_kernel_size=strided_kernel_size,
                    stride=stride,
                    groups=groups,
                    normalization=normalization,
                    activation=activation,
                    out_activation=activation,
                    dropout=dropout,
                    dim=dim
                ))

        self.res_convs = nn.Sequential(*self.res_convs)

        embed_dim = self.base_filters*c_mult[-1]
        if class_attention_block is not None:
            self.final_pool = class_attention_block(
                in_features=embed_dim)
            self.cls_token = nn.Parameter(torch.zeros((1, 1, embed_dim)))
            self.last_norm = nn.LayerNorm(embed_dim)
            nn.init.normal_(self.cls_token, std=.02)
        else:
            self.final_pool = utils.get_adaptive_averagepool_nd(dim)(
                output_size=[1 for i in range(dim)])

        self.fc = nn.Linear(
            in_features=embed_dim,
            out_features=out_channels) if out_channels != 0 else nn.Identity()

        self.act = out_activation() if out_activation is not None else None

        self.apply(self._init_weights)

        self.rel_position_input = rel_position_input

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weigh_decay(self) -> set:
        return {'cls_token'}

    def cat_rel_pos(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate relative position to input."""
        rel_pos = []
        for d in range(x.ndim-2):
            rel_pos.append(torch.arange(
                x.shape[d+2], device=x.device, dtype=x.dtype)/(x.shape[d+2]-1))

        rel_pos = torch.stack(torch.meshgrid(*rel_pos), dim=0)[None]
        rel_pos = rel_pos.repeat(
            x.shape[0],
            *[1 for i in range(rel_pos.ndim-1)]
        )
        x = torch.cat((x, rel_pos), dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rel_position_input:
            x = self.cat_rel_pos(x)
        x = self.input_conv(x)
        x = self.pool(x)
        x = self.res_convs(x)
        if isinstance(self.final_pool, transformers.ClassAttentionBlock):
            x = x.flatten(2, -1).transpose(1, 2)  # BxNxE
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # BxN+1xE
        x = self.final_pool(x)
        if isinstance(self.final_pool, transformers.ClassAttentionBlock):
            x = self.last_norm(x)[:, 0]
        x = x.flatten(1, -1)
        x = self.fc(x)
        if self.act is not None:
            x = self.act(x)
        return x
