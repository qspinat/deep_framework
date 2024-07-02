"""lightning trainer class."""

import gin
import lightning as L
import numpy as np
from torch.utils import data

from deep_framework import base_dataset


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
        if weight_positives:
            ds = example_ds()
            labels = ds.csv_dataset.df.loc[
                ds.uids, ds.csv_dataset.target_features[0]].values
            lab, counts = np.unique(labels, return_counts=True)
            print("LAB COUNT", lab, counts)
            weights = np.ones_like(labels)
            for l, c in zip(lab, counts):
                weights[labels == l] = 1/c
            sampler = data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(labels),
                replacement=True,
            )
        else:
            sampler = None
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
