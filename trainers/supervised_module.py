"""Supervised lightning module class"""

import functools
from typing import Any, Callable, Sequence

import gin
from kornia import augmentation
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch import optim
import torchmetrics as tm


@gin.register(module="trainers")
class SupervisedLModule(L.LightningModule):
    """Supervised lightning module class. Used to train supervised models."""

    def __init__(
            self,
            model: type[nn.Module],
            losses: Sequence[tuple[type[nn.Module], float]],
            metrics: Sequence[type[nn.Module]],
            torchmetrics: Sequence[type[tm.Metric]],
            input_label: str,
            target_label: str,
            *args,
            tabular_label: str = "tabular",
            weight_label: str = "weight",
            optimizer: type[optim.Optimizer] | None,
            lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler],
            data_aug_gpu: Sequence[augmentation.AugmentationBase2D |
                                   augmentation.AugmentationBase3D] | None = None,
            data_aug_batch_size: int = -1,
            activation_metric: Callable = functools.partial(
                torch.softmax, dim=-1),
            flatten_target: bool = False,
            **kwargs,
    ) -> None:
        """Constructor.

        Args:
            model (torch.nn.Module): Model to train.
            losses (Sequence[tuple[torch.nn.Module, float]]): Losses to use.
            metrics (Sequence[torch.nn.Module]): Metrics to use.
            torchmetrics (Sequence[torchmetrics.Metric]): Torchmetrics to use.
            input_label (str): Input label.
            target_label (str): Target label.
            tabular_label (str): Tabular label. Default to None.
            weight_label (str): Weight label. Default to None.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler to use.
            data_aug_gpu (Sequence[kornia.AugmentationBase2D | kornia.AugmentationBase2D]): 
                Data augmentation to use on GPU.
            data_aug_batch_size (int): Data augmentation batch size. If less than 0, the batch size is the 
                same as the input batch size. Default to -1.
            activation_metric (callable): Activation to use before metric. Default to softmax.
            flatten_target (bool): Whether to flatten the target or not. Default to False.
        """
        super().__init__(*args, **kwargs)
        self.model = model()
        self.loss_names = []
        self.loss_weights = []
        for l, w in losses:
            l = l()
            name = (l.name if hasattr(l, "name") else l.__class__.__name__)
            self.__setattr__(name, l)
            self.loss_weights.append(w)
            self.loss_names.append(name)
        self.metric_names = []
        for m in metrics:
            m = m()
            name = (m.name if hasattr(m, "name") else m.__class__.__name__)
            self.__setattr__(name, m)
            self.metric_names.append(name)
        self.torchmetric_names = []
        for m in torchmetrics:
            val_m = m()
            train_m = m()
            val_name = (val_m.name if hasattr(val_m, "name")
                        else val_m.__class__.__name__)
            train_name = (train_m.name if hasattr(train_m, "name")
                          else train_m.__class__.__name__)
            self.__setattr__(f"val_{val_name}", val_m)
            self.__setattr__(f"train_{train_name}", train_m)
            self.torchmetric_names.append(val_name)
        self.input_label = input_label
        self.target_label = target_label
        self.weight_label = weight_label
        self.tabular_label = tabular_label
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        data_aug_gpu = [] if data_aug_gpu is None else data_aug_gpu
        self.data_aug_gpu = nn.Sequential(*data_aug_gpu)
        self.data_aug_batch_size = data_aug_batch_size
        self.act_metric = activation_metric
        self.flatten_target = flatten_target
        self.save_hyperparameters(ignore=["data_aug_gpu"])

    @torch.no_grad()
    def on_after_batch_transfer(
        self,
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

    def compute_and_log_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        log_label: str
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            pred (torch.Tensor): Predictions.
            target (torch.Tensor): Targets.
            log_label (str): Log label.

        Returns:
            torch.Tensor: Loss.
        """
        loss = torch.zeros(1, dtype=pred.dtype, device=pred.device)

        def log_fn(name: str, value: float):
            self.log(
                f"{log_label}.{name}",
                value,
                on_epoch=True,
                on_step=False,
            )
        for l_name, w in zip(self.loss_names, self.loss_weights):
            l = self.__getattr__(l_name)
            _loss = l(pred, target, weight=weight, log_fn=log_fn)
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
        return loss

    def compute_and_log_metrics(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            log_label: str
    ) -> None:
        """Compute and log metrics.

        Args:
            pred (torch.Tensor): Predictions.
            target (torch.Tensor): Targets.
            log_label (str): Log label.
        """
        for m_name in self.metric_names:
            m = self.__getattr__(m_name)
            self.log(
                f"{log_label}.{m_name}",
                m(self.act_metric(pred), target).item(),
                on_epoch=True,
                on_step=False,
            )
        for m_name in self.torchmetric_names:
            m: tm.Metric = self.__getattr__(
                f"{log_label}_{m_name}")
            m.update(self.act_metric(pred), target)
            self.log(f'{log_label}.{m_name}',
                     m,
                     on_epoch=True,
                     on_step=False)

    def base_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        log_label: str,
        *args,
        **kwargs,
    ) -> STEP_OUTPUT:
        x = batch[self.input_label]
        target = batch[self.target_label]
        weight = (batch[self.weight_label]
                  if self.weight_label in batch else None)
        inputs = [x]
        if self.tabular_label in batch:
            inputs.append(batch[self.tabular_label])
        if self.flatten_target:
            target = target.flatten()
        pred = self.model(*inputs)
        loss = self.compute_and_log_losses(
            pred=pred,
            target=target,
            weight=weight,
            log_label=log_label
        )
        self.compute_and_log_metrics(
            pred=pred,
            target=target,
            log_label=log_label
        )
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
