"""Swin Transformer model for image classification. Inspired from 
monai implementation of SwinUnetr."""

from typing import Sequence

import einops
import gin
from monai.networks.nets import swin_unetr
from monai.networks.blocks import PatchEmbed, UnetrBasicBlock
from monai.utils import look_up_option
import torch
from torch import nn
import torch.nn.functional as F

MERGING_MODE = {"merging": swin_unetr.PatchMerging,
                "mergingv2": swin_unetr.PatchMergingV2}


@gin.register(module="models")
class SwinT(nn.Module):
    """
    Args:
        in_chans: dimension of input channels.
        embed_dim: number of linear projection output channels.
        window_size: local window size.
        patch_size: patch size.
        depths: number of layers in each stage.
        num_heads: number of attention heads.
        mlp_ratio: ratio of mlp hidden dim to embedding dim.
        qkv_bias: add a learnable bias to query, key, value.
        drop_rate: dropout rate.
        attn_drop_rate: attention dropout rate.
        drop_path_rate: stochastic depth rate.
        norm_layer: normalization layer.
        patch_norm: add normalization after patch embedding.
        use_checkpoint: use gradient checkpointing for reduced memory usage.
        spatial_dims: spatial dimension.
        downsample: module used for downsampling, available options are
            `"mergingv2"`, `"merging"` and a user-specified `nn.Module`
            following the API defined in
            :py:class:`monai.networks.nets.PatchMerging`. The default is
            currently `"merging"` (the original version defined in v0.9.0).
        use_v2: using swinunetr_v2, which adds a residual convolution block at
            the beginning of each swin stage.
        """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = -1,
        embed_dim: int = 24,
        window_size: Sequence[int] = (4, 4, 4),
        patch_size: Sequence[int] = (4, 4, 4),
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str = "mergingv2",
        use_v2: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths))]
        self.use_v2 = use_v2

        down_sample_mod = (look_up_option(downsample, MERGING_MODE)
                           if isinstance(downsample, str) else downsample)

        self.layers = []
        if use_v2:
            self.layersc = []

        for i_layer in range(self.num_layers):
            layer = swin_unetr.BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer])
                                  : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if self.use_v2:
                layerc = UnetrBasicBlock(
                    spatial_dims=3,
                    in_channels=embed_dim * 2**i_layer,
                    out_channels=embed_dim * 2**i_layer,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                self.layersc.append(layerc)
        self.layers = nn.ModuleList(self.layers)
        self.layersc = nn.ModuleList(self.layersc)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_size = patch_size
        self.embed_dim = int(embed_dim * 2 ** self.num_layers)

        if out_channels > 0:
            self.fc = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.embed_dim, out_channels)
            )

        else:
            self.fc = nn.Identity()

    def proj_out(self, x: torch.Tensor, normalize=False) -> torch.Tensor:
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = einops.rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = einops.rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = einops.rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = einops.rearrange(x, "n h w c -> n c h w")
        return x

    def get_intermediate_layers(
        self, x: torch.Tensor, normalize: bool = True
    ) -> list[torch.Tensor]:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        outs = [self.proj_out(x, normalize)]
        for i in range(self.num_layers):
            if self.use_v2:
                x = self.layersc[i](x.contiguous())
            x = self.layers[i](x.contiguous())
            outs.append(self.proj_out(x, normalize))
        return outs

    def forward_features(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            if self.use_v2:
                x = self.layersc[i](x.contiguous())
            x = self.layers[i](x.contiguous())
        x = self.proj_out(x, True)
        x = x.flatten(2).mean(2)
        return {
            "x_norm_clstoken": x,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)["x_norm_clstoken"]
        return self.fc(x)
