"""Transformer blocks."""

import gin
import torch
from torch import nn

from . import dropout
from . import mlp


class ClassAttention(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: bool = None,
        attn_drop_rate: float = 0.,
        proj_drop_rate: float = 0.
    ):
        """Constructor.

        Args:
            in_features (int): Number of input features.
            num_heads (int): Number of attention heads. Default to 8.
            qkv_bias (bool): Whether to use bias in qkv linear layers. Default to False.
            qk_scale (bool): Whether to scale qk. Default to None.
            attn_drop_rate (float): Attention dropout rate. Default to 0.
            proj_drop_rate (float): Projection dropout rate. Default to 0.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = in_features // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(in_features, in_features * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(in_features, in_features)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(
            self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, N, E = x.shape
        # B, N, E*3 --> B, N, 3, num_heads, E//num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, E // self.num_heads)
        # B, N, 3, num_heads, E//num_heads --> 3, B, num_heads, N, E//num_heads
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # B, num_heads, N, E//num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]

        qc = q[:, :, 0:1]   # CLS token
        attn_cls_ = (qc * k).sum(dim=-1) * self.scale
        attn_cls = self.attn_drop(attn_cls_)
        attn_cls = attn_cls.softmax(dim=-1)

        cls_tkn = (attn_cls.unsqueeze(2) @ v).transpose(1, 2).reshape(B, 1, E)
        cls_tkn = self.proj(cls_tkn)
        x = torch.cat([self.proj_drop(cls_tkn), x[:, 1:]], dim=1)
        if return_attention:
            return x, attn_cls_
        return x


@gin.configurable(module="blocks")
class ClassAttentionLayer(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout_rate: float = 0.,
        attn_dropout_rate: float = 0.,
        dropout_path_rate: float = 0.,
        activation: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        eta: float | None = None,
        tokens_norm: bool = False
    ):
        """Constructor.

        Args:
            in_features (int): Number of input features.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Ratio of hidden features in mlp. Default to 4.
            qkv_bias (bool): Whether to use bias in qkv linear layers. Default to False.
            qk_scale (float | None): Scale of qk. Default to None.
            dropout_rate (float): Dropout rate. Default to 0.
            attn_dropout_rate (float): Attention dropout rate. Default to 0.
            dropout_path_rate (float): Dropout path rate. Default to 0.
            activation (torch.nn.Module): Activation layer. Default to GELU.
            norm_layer (torch.nn.Module): Normalization layer. Default to LayerNorm.
            eta (float | None): LayerScale Initialization. Default to None.
            tokens_norm (bool): Whether to normalize tokens. Default to False.
        """
        super().__init__()
        self.norm1 = norm_layer(in_features)

        self.attn = ClassAttention(
            in_features=in_features,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_dropout_rate,
            proj_drop_rate=dropout_rate
        )

        self.drop_path = dropout.DropPath(
            dropout_path_rate) if dropout_path_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(in_features)
        mlp_hidden_dim = int(in_features * mlp_ratio)
        self.mlp = mlp.MLP(
            in_features=in_features,
            hidden_features=[mlp_hidden_dim],
            out_features=in_features,
            activation=activation,
            dropout=nn.Dropout(dropout_rate))

        # LayerScale Initialization (no layerscale when None)
        if eta is not None:
            self.gamma1 = nn.Parameter(
                eta * torch.ones(in_features), requires_grad=True)
            self.gamma2 = nn.Parameter(
                eta * torch.ones(in_features), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        self.tokens_norm = tokens_norm

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attention:
            x_, attn = self.attn(self.norm1(x), return_attention)
        else:
            x_ = self.attn(self.norm1(x))
        x = x + self.drop_path(self.gamma1 * x_)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = torch.cat((self.norm2(x[:, 0:1]), x[:, 1:]), dim=1)
        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        if return_attention:
            return x, attn
        return x


@gin.register(module="blocks")
class ClassAttentionBlock(nn.Module):
    """Class Attention Block as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(
        self,
        in_features: int,
        class_attention_layer: type[ClassAttentionLayer],
        n_layers: int,
        normalization: type[nn.Module] = nn.LayerNorm,
        activation: type[nn.Module] = nn.ReLU,
        *args,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            in_features (int): Number of input features.
            class_attention_layer (type[ClassAttentionLayer]): Class Attention Layer.
            n_layers (int): Number of layers.
            normalization (torch.nn.Module): Normalization layer. Default to LayerNorm.
            activation (torch.nn.Module): Activation layer. Default to ReLU.
        """
        super().__init__(*args, **kwargs)
        layers = []
        for i in range(n_layers):
            layers.append(class_attention_layer())
            if i < n_layers-1:
                layers.append(normalization(in_features))
                layers.append(activation())
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, return_attention: bool
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        attns = []
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, True)
                attns.append(attn)
            else:
                x = layer(x, False)
        if return_attention:
            return x, attns
        return x
