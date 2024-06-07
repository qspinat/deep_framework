""" Patch Aggregator module. """

import functools

import gin
import torch
from torch import nn

from ..blocks import transformers


@gin.register(module="models")
class PatchAggregator(nn.Module):
    """Patch Aggregator. Applies a backbone to a set of patches and aggregates the results."""

    def __init__(self,
                 backbone: type[nn.Module],
                 embed_dim: int,
                 out_features: int,
                 pooling_cls: (
                     type[nn.AdaptiveAvgPool1d] |
                     type[nn.AdaptiveMaxPool1d] |
                     type[transformers.ClassAttentionBlock]
                 ) = functools.partial(nn.AdaptiveMaxPool1d, output_size=1),
                 max_batch_size: int = -1,
                 freeze_backbone: bool = True,
                 *args,
                 **kwargs) -> None:
        """Constructor.

        Args:
            backbone (torch.nn.Module): Backbone model.
            embed_dim (int): Embedding dimension.
            out_features (int): Number of output features.
            pooling_cls (torch.nn.Module): Pooling layer. Default to AdaptiveMaxPool1d.
            max_batch_size (int): Maximum batch size. Default to -1.
            freeze_backbone (bool): Whether to freeze the backbone or not. Default to True.
        """
        super().__init__(*args, **kwargs)
        self.max_batch_size = max_batch_size
        self.pool = pooling_cls()
        self.last_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(in_features=embed_dim, out_features=out_features)
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone()
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if isinstance(self.pool, transformers.ClassAttentionBlock):
            self.cls_token = nn.Parameter(torch.zeros((1, 1, embed_dim)))
            self.last_norm = nn.LayerNorm(embed_dim)
            nn.init.normal_(self.cls_token, std=.02)

    def predict_features(self, x: torch.Tensor) -> torch.Tensor:
        """Predict features from input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted features.
        """
        if self.max_batch_size <= 0:
            return self.backbone(x)
        B = x.shape[0]
        features = []
        n_preds = B//self.max_batch_size
        for i in range(n_preds):
            features.append(self.backbone(x[
                i*self.max_batch_size:(i+1)*self.max_batch_size]))
        if B % self.max_batch_size != 0:
            features.append(self.backbone(x[n_preds*self.max_batch_size:]))
        features = torch.cat(features, dim=0)
        return features

    def aggregate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate features.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Aggregated features.
        """
        x = x.unsqueeze(0)
        if isinstance(self.pool, transformers.ClassAttentionBlock):
            x = torch.cat((self.cls_token, x), dim=1)  # 1xB+1xE
            x = self.pool(x)[:, 1, :]  # 1xE
        else:
            x = x.transpose(1, 2)  # 1xExB
            x = self.pool(x)  # 1xE
        x = x.flatten(1, -1)  # 1xE
        x = self.last_norm(x)
        return self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.predict_features(x)
        x = self.aggregate_features(x)
        return x
