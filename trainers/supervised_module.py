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


@gin.register(module="trainers")
class SupervisedLModule(L.LightningModule):
    def __init__(
            self,
            model: type[nn.Module],
            losses: Sequence[tuple[type[nn.Module], float]],
            metrics: Sequence[type[nn.Module]],
            torchmetrics: Sequence[type[torchmetrics.Metric]],
            input_label: str,
            target_label: str,
            optimizer: type[optim.Optimizer] | None,
            lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler],
            data_aug_gpu: Sequence[augmentation.AugmentationBase2D] | None = None,
            data_aug_batch_size: int = -1,
            activation_metric: callable = functools.partial(
                torch.softmax, dim=-1),
            flatten_target: bool = False,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model()
        self.loss_names = []
        self.loss_weights = []
        for l, w in losses:
            l = l()
            self.__setattr__(l.__class__.__name__, l)
            self.loss_weights.append(w)
            self.loss_names.append(l.__class__.__name__)
        self.metric_names = []
        for m in metrics:
            m = m()
            self.__setattr__(m.__class__.__name__, m)
            self.metric_names.append(m.__class__.__name__)
        self.torchmetric_names = []
        for m in torchmetrics:
            val_m = m()
            train_m = m()
            self.__setattr__(f"val_{val_m.__class__.__name__}", val_m)
            self.__setattr__(f"train_{train_m.__class__.__name__}", train_m)
            self.torchmetric_names.append(val_m.__class__.__name__)
        self.input_label = input_label
        self.target_label = target_label
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        data_aug_gpu = [] if data_aug_gpu is None else data_aug_gpu
        self.data_aug_gpu = nn.Sequential(*data_aug_gpu)
        self.data_aug_batch_size = data_aug_batch_size
        self.act_metric = activation_metric
        self.flatten_target = flatten_target

    def on_after_batch_transfer(self,
                                batch: dict[str, torch.Tensor],
                                dataloader_idx: int
                                ) -> dict[str, torch.Tensor]:
        if self.trainer.training:
            if self.data_aug_batch_size <= 0:
                batch[self.input_label] = self.data_aug_gpu(
                    batch[self.input_label])
            else:
                B = batch[self.input_label].shape[0]
                bs = self.data_aug_batch_size
                n_preds = B//bs
                for i in range(n_preds):
                    batch[self.input_label][i*bs:(i+1)*bs] = self.data_aug_gpu(
                        batch[self.input_label][i*bs:(i+1)*bs])
                if B % bs != 0:
                    batch[self.input_label][n_preds*bs:] = self.data_aug_gpu(
                        batch[self.input_label][n_preds*bs:])
        return batch

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
        if self.flatten_target:
            target = batch[self.target_label].flatten()
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

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        *args,
        **kwargs,
    ) -> STEP_OUTPUT:
        loss = self.base_step(batch, batch_idx, "train", *args, **kwargs)
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        *args: Any,
        **kwargs: Any
    ) -> STEP_OUTPUT:
        loss = self.base_step(batch, batch_idx, "val", *args, **kwargs)
        return loss

    def on_validation_end(self) -> None:
        torch.cuda.empty_cache()

    def on_train_end(self) -> None:
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = self.optimizer(
            filter(lambda p: p.requires_grad, self.parameters()))
        if self.lr_scheduler is None:
            return optimizer
        lr_scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
