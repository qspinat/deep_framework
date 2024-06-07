"""Wrapper for timm models."""

from typing import Any

import gin
import timm
import torch
from torch import nn


@gin.register(module="models")
class TimmModel(nn.Module):
    """Wrapper for timm models."""

    def __init__(
        self,
        model_name: str,
        model_cfg: dict[str, Any],
        *args,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            model_name (str): Model name.
            model_cfg (dict[str, Any]): Model configuration.
        """
        if model_name not in timm.list_models():
            raise ValueError(
                f"{model_name} not in timm model list: {timm.list_models()}")
        super().__init__(*args, **kwargs)
        self.model = timm.create_model(
            model_name=model_name,
            **model_cfg
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
