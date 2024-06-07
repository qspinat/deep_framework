"""Supervised lightning module class"""

import functools
from typing import Any, Sequence

import gin
from kornia import augmentation
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch import optim
import torchmetrics

from .supervised_module import SupervisedLModule


@gin.register(module="trainers")
class SegmentationLModule(SupervisedLModule):

    def base_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        log_label: str,
        *args,
        **kwargs,
    ) -> STEP_OUTPUT:
        x = batch[self.input_label]
        device = x.device
        target = batch[self.target_label]
        pred = self.model(x)
        loss = torch.zeros(1, dtype=x.dtype, device=device)
        for l_name, w in zip(self.loss_names, self.loss_weights):
            l = self.__getattr__(l_name)
            _loss = l(pred, target)
            self.log(
                f"{log_label}.{l_name}",
                _loss.item(),
                on_epoch=True,
                on_step=False,
            )
            loss += w*_loss
        self.log(f"{log_label}.loss",
                 loss.item(),
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True)
        for m_name in self.metric_names:
            m = self.__getattr__(m_name)
            self.log(
                f"{log_label}.{m_name}",
                m(self.act_metric(pred), target).item(),
                on_epoch=True,
                on_step=False,
            )
        for m_name in self.torchmetric_names:
            m = self.__getattr__(f"{log_label}_{m_name}")
            m.update(self.act_metric(pred), target)
            self.log(f'{log_label}.{m_name}',
                     m,
                     on_epoch=True,
                     on_step=False)
        return loss
