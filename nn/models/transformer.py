""" Transformer model implementation. """

import gin
import torch
from torch import nn

from ..blocks import mlp
from ..blocks import transformers


@gin.register(module="models")
class Transformer(nn.Module):
    """Transformer model implementation."""

    def __init__(
        self,
        in_features: int,
        transformer_block: type[nn.TransformerEncoder] | None,
        class_attention_block: type[transformers.ClassAttentionBlock] | None,
        mlp: type[mlp.MLP],
        normalization: type[nn.Module] = nn.LayerNorm,
        activation: type[nn.Module] = nn.ReLU,
        *args,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            in_features (int): Number of input features.
            transformer_block (type[nn.TransformerEncoder]): Transformer block. Default to None.
            class_attention_block (type[transformers.ClassAttentionBlock]): Class Attention Block. 
                Default to None.
            mlp (type[mlp.MLP]): Multi Layer Perceptron.
            normalization (torch.nn.Module): Normalization layer. Default to LayerNorm.
            activation (torch.nn.Module): Activation layer. Default to ReLU.
        """
        super().__init__(*args, **kwargs)
        transformer_sequence = []
        if transformer_block is not None:
            transformer_sequence.extend([
                transformer_block(),
                normalization(in_features),
                activation(),
            ])
        cls_attention_sequence = []
        if class_attention_block is not None:
            cls_attention_sequence.append(
                nn.ModuleList([
                    class_attention_block(),
                    normalization(in_features),
                ]))
        self.transformer_block = nn.Sequential(*transformer_sequence)
        self.class_attention_block = nn.ModuleList(cls_attention_sequence)
        self.mlp = mlp()
        self.cls_token = nn.Parameter(torch.zeros((1, 1, in_features)))
        nn.init.normal_(self.cls_token, std=.02)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.transformer_block(x)  # BxNxE
        x = torch.cat((self.cls_token, x), dim=1)
        for ca_layer in self.class_attention_block:
            x = ca_layer[0](x, return_attention)
            if return_attention:
                x, attns = x
            x = ca_layer[1](x)
        x = x[:, 0]  # cls_token # BxE
        x = self.mlp(x)
        if return_attention:
            return x, attns
        return x
