"""Transformer blocks."""

import abc
import math
from typing import Sequence

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


@gin.register(module="blocks")
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
        self, x: torch.Tensor, return_attention: bool = False,
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
        self, x: torch.Tensor, return_attention: bool = False,
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


class PositionEncoding(nn.Module, abc.ABC):
    """Base class for all transformer input sequence position encoders."""

    def __init__(self,
                 projection_size: int,
                 sequence_length: int | Sequence[int],
                 encode_cls_token: bool,
                 **kwargs):
        """Inits PositionEncoding.

        Args:
            projection_size (int): number of features in the linear projected
                representation.
            sequence_length (int | Sequence[int]): the number of elements
                in the sequential input. It can either be a single int denoting
                a 1d or flattened sequence or a tuple of ints denoting the
                sequence length in each direction.
            encode_cls_token (bool): Whether to add a position encoding for
                the cls_token or not.
        """
        super().__init__(**kwargs)
        self.projection_size = projection_size
        self.sequence_length = sequence_length
        self.encode_cls_token = encode_cls_token

    @property
    @abc.abstractmethod
    def encoding(self):
        ...

    def _cat_cls_token(self,
                       projection: torch.Tensor,
                       cls_token: torch.Tensor | None) -> torch.Tensor:
        if cls_token is None:
            return projection
        b = projection.shape[0]
        cls_token = cls_token.expand(b, *cls_token.shape)
        return torch.cat([cls_token, projection], dim=1)

    def forward(self,
                input: torch.Tensor,
                cls_token: torch.Tensor | None) -> torch.Tensor:
        """

        Args:
            input (torch.Tensor<float>[B,N,E]): linear projection of the input.
                Where E is the projection_size (embedding size), and N is the
                total number of patches in the flattened sequence.
            cls_token (Optional[torch.Tensor<float>[1,E]]): the learnable
                classification token.

        Returns:
            Union[torch.Tensor<float>[B,N+1,E],torch.Tensor<float>[B,N,E]]:
                position encoded input with concatenated cls_token if not
                none. Where E is the projection_size (embedding size).
        """
        if self.encode_cls_token:
            return self._cat_cls_token(input, cls_token) + self.encoding
        return self._cat_cls_token(input + self.encoding, cls_token)


@gin.register(module="blocks")
class LearntPositionEncoding(PositionEncoding):
    """Position encoder where encodings are learnt during training.

    This encoder is the same as the one used in the original BERT and ViT
    papers.
    """

    def __init__(self,
                 projection_size: int,
                 sequence_length: int | Sequence[int],
                 encode_cls_token: bool = True,
                 **kwargs):
        super().__init__(projection_size,
                         sequence_length,
                         encode_cls_token,
                         **kwargs)
        self._encoding = self.make_learnable_encoding()

    def make_learnable_encoding(self) -> torch.Tensor:
        sequence_length = (self.sequence_length
                           if isinstance(self.sequence_length, int)
                           else torch.tensor(self.sequence_length).prod().int())
        sequence_length = (sequence_length + 1
                           if self.encode_cls_token else sequence_length)
        encoding = nn.Parameter(torch.empty(1,
                                            sequence_length,
                                            self.projection_size))
        # the value of 0.02 for std comes from the official jax impl of ViT
        # https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py#L187
        # which inturn come from BERT.
        nn.init.normal_(encoding, std=0.02)
        return encoding

    @property
    def encoding(self):
        return self._encoding


def _sincos_pos_encoding_1d(projection_size: int,
                            sequence_length: int,
                            temperature: float = 10000.0) -> torch.Tensor:
    positions = torch.arange(0, sequence_length).unsqueeze(1)
    encoding = torch.empty(1, sequence_length, projection_size)
    div_term = torch.exp(torch.arange(
        0, projection_size, 2) * (-math.log(temperature) / projection_size))
    encoding[0, :, 0::2] = torch.sin(positions * div_term)
    encoding[0, :, 1::2] = torch.cos(positions * div_term)
    return encoding


@gin.register(module="blocks")
class NdSinCosPositionEncoding(PositionEncoding):
    """Position encoder using multiple sine waves of different frequencies.

    This encoder is the same as one used in the original transformer paper,
    Ashish Vaswani et al. "Attention Is All You Need", 2017
    (https://arxiv.org/pdf/1706.03762.pdf), if the input to `sequence_length`
    is an integer or a 1 element list.

    If the input to `sequence_length` is a list with `n` integers then it
    encodes an nd-sin-cos embedding which is a generalization of the 2d
    sin-cos encoding introduced for images in MoCo v3, Xinlei Chen et al.
    "An Empirical Study of Training Self-Supervised Vision Transformers", 2021.
    (https://arxiv.org/pdf/2104.02057.pdf)
    """

    def __init__(self,
                 projection_size: int,
                 sequence_length: int | Sequence[int],
                 encode_cls_token: bool = False,
                 temperature: float = 10000.0,
                 **kwargs):
        """Inits NdSinCosPositionEncoding.

        Args:
            temperature (float): In sin-cos encoding the wavelengths form a
                geometric progression from 2π to temperature · 2π. Therefore,
                the parameter temperature controls the maximum wavelength
                of the sin/cos waves. Defaults to 10000.0.

        Raises:
            ValueError: If projection_size cannot be split in equal parts
                to encode all directions, i.e., projection_size must be
                divisible by twice the number of dimensions in data.
            ValueError: If `encode_cls_token` is set to `True`.
        """
        if encode_cls_token:
            raise ValueError(
                f"{self.__class__.__name__} does not support"
                " encoding of classification token.")
        sequence_length = ([sequence_length]
                           if isinstance(sequence_length, int)
                           else sequence_length)
        if projection_size % (2 * len(sequence_length)):
            raise ValueError(
                "Expected projection size to be divisible by "
                f"{2*len(sequence_length)} in order to use sin-cos position"
                f" embedding. Found projection size {projection_size}"
                " instead.")
        self.temperature = temperature
        super().__init__(projection_size,
                         sequence_length,
                         encode_cls_token,
                         **kwargs)
        encoding = self.make_nd_sincos_encoding()
        self.register_buffer("_encoding", encoding)
        self._encoding: torch.Tensor

    @property
    def encoding(self):
        return self._encoding

    def make_nd_sincos_encoding(self) -> torch.Tensor:
        num_dims = len(self.sequence_length)
        projection_size_per_dim = int(self.projection_size / num_dims)
        partial_encodings = [
            _sincos_pos_encoding_1d(
                projection_size_per_dim, seq_len, self.temperature)
            for seq_len in self.sequence_length]
        final_encoding = torch.empty(
            1, *self.sequence_length, self.projection_size)
        for i, encoding in enumerate(partial_encodings):
            encoding = encoding.view(
                1, encoding.shape[1], *[1] * (num_dims - 1), encoding.shape[2]
            ).movedim(1, i + 1)
            proj_indices_st = i * projection_size_per_dim
            proj_indices_end = proj_indices_st + projection_size_per_dim
            final_encoding[..., proj_indices_st:proj_indices_end] = encoding
        return final_encoding.view(1, -1, self.projection_size)


@gin.register(module="blocks")
class NdSinCosLearntPositionEncoding(NdSinCosPositionEncoding):
    """
    Positional encoding relying on a fourier kernel and a learnt embedding
    matching the one used in Ashish Vaswani et al. "Attention Is All You Need",
    2017 (https://arxiv.org/pdf/1706.03762.pdf).
    """

    def __init__(
            self,
            projection_size: int,
            sequence_length: int | Sequence[int],
            hidden_dim: int = 32,
            temperature: float = 10000,
            dim: int = 3,
            **kwargs):
        """Inits NdSinCosLearntPositionEncoding.

        Args:
            projection_size (int): number of features in the linear projected
                representation.
            sequence_length (Union[int, Sequence[int]]): the number of elements
                in the sequential input. It can either be a single int denoting
                a 1d or flattened sequence or a tuple of ints denoting the
                sequence length in each direction.
            hidden_dim (int): Hidden dimension in which each dimension position
                is encoded using multiple sine waves of different frequencies.
                Defaults to 32.
            temperature (float): In sin-cos encoding the wavelengths form a
                geometric progression from 2π to temperature · 2π. Therefore,
                the parameter temperature controls the maximum wavelength
                of the sin/cos waves. Defaults to 10000.0.
            dim (int): Dimension of the input for 1d, 2d or 3d. Defaults to 3.
        """
        if dim is None and isinstance(sequence_length, int):
            raise ValueError(
                "dim must be specified if sequence_length is an int")
        dim = dim or len(sequence_length)
        if isinstance(sequence_length, int):
            sequence_length = [sequence_length] * dim
        if len(sequence_length) != dim:
            raise ValueError("dim != len(sequence_length)")

        super().__init__(projection_size=hidden_dim * dim,
                         sequence_length=sequence_length,
                         encode_cls_token=False,
                         temperature=temperature,
                         **kwargs)
        self.dim = dim
        self.token_projection = nn.Linear(
            hidden_dim * dim, projection_size)

    def forward(self,
                input: torch.Tensor,
                cls_token: torch.Tensor | None = None) -> torch.Tensor:
        """

        Args:
            input (torch.Tensor<float>[B,N,E]): linear projection of the input.
                Where E is the projection_size (embedding size), and N is the
                total number of patches in the flattened sequence.
            cls_token (Optional[torch.Tensor<float>[1,E]]): the learnable
                classification token.

        Returns:
            Union[torch.Tensor<float>[B,N+1,E],torch.Tensor<float>[B,N,E]]:
                position encoded input with concatenated cls_token if not
                none. Where E is the projection_size (embedding size).
        """
        if self.encode_cls_token:
            return (self._cat_cls_token(input, cls_token) +
                    self.token_projection(self.encoding))
        return self._cat_cls_token(
            input + self.token_projection(self.encoding), cls_token)
