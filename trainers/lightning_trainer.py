"""lightning trainer class."""

import gin
import lightning as L
from torch.utils import data

from deep_framework.datasets import base_dataset
from deep_framework.trainers import utils


@gin.configurable(module="trainers")
class Trainer:
    """Trainer class for lightning models.

    Attributes:
        lightning_module (type[L.LightningModule]): Lightning module.
        lightning_trainer (type[L.Trainer]): Lightning trainer.
        train_loader (type[data.DataLoader]): Train dataloader.
        val_loader (type[data.DataLoader] | None): Validation dataloader.
        ckpt_path (str | None): Checkpoint path.
        seed (int): Seed.
    """

    def __init__(
        self,
        lightning_module: type[L.LightningModule],
        lightning_trainer: type[L.Trainer],
        train_loader: type[data.DataLoader],
        val_loader: type[data.DataLoader] | None = None,
        ckpt_path: str | None = None,
        weight_positives: bool = False,
        example_ds: type[base_dataset.BaseDataset] = None,
        seed: int = 42,
    ) -> None:
        """Constructor.

        Args:
            lightning_module (type[L.LightningModule]): Lightning module.
            lightning_trainer (type[L.Trainer]): Lightning trainer.
            train_loader (type[data.DataLoader]): Train dataloader.
            val_loader (type[data.DataLoader] | None): Validation dataloader.
                Default to None.
            ckpt_path (str | None): Checkpoint path. Default to None.
            weight_positives (bool): Whether to weight positives or not. 
                Default to False.
            example_ds (type[base_dataset.BaseDataset]): Dataset from which to 
                sample the weight for weighting positives.
            seed (int): Seed. Default to 42.
        """
        L.seed_everything(seed)
        sampler = (utils.get_weighted_sampler(example_ds())
                   if weight_positives else None)
        self.lightning_module = lightning_module()
        self.lighting_trainer = lightning_trainer()
        self.train_loader = train_loader(sampler=sampler)
        self.val_loader = val_loader() if val_loader is not None else None
        self.ckpt_path = ckpt_path
        self.seed = seed

    def fit(self) -> None:
        """Fit the model."""
        L.seed_everything(self.seed)
        self.lighting_trainer.fit(
            model=self.lightning_module,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=self.ckpt_path,
        )
