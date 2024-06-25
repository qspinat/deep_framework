"""lightning trainer class."""

import os
from typing import Sequence

import gin
import pandas as pd
from torch.utils import data
import lightning as L
from lightning.pytorch import loggers as L_loggers


@gin.configurable(module="trainers")
class KFoldTrainer:
    """Trainer class for lightning models.

    Attributes:
        lightning_module (type[L.LightningModule]): Lightning module.
        lightning_trainer (type[L.Trainer]): Lightning trainer.
        train_dataset (type[data.Dataset]): Train dataset.
        train_loader (type[data.DataLoader]): Train dataloader.
        val_dataset (type[data.Dataset] | None): Validation dataset.
        val_loader (type[data.DataLoader] | None): Validation dataloader.
        ckpt_path (str | None): Checkpoint path.
        seed (int): Seed.
    """

    def __init__(
        self,
        root_dir: str,
        k_folds: int,
        lightning_module: type[L.LightningModule],
        lightning_trainer: type[L.Trainer],
        loggers: Sequence[type[L_loggers.Logger]],
        train_patients_txts: Sequence[str],
        val_patients_txts: Sequence[str],
        train_dataset: type[data.Dataset],
        train_loader: type[data.DataLoader],
        val_dataset: type[data.Dataset],
        val_loader: type[data.DataLoader],
        ckpt_path: str | None = None,
        seed: int = 42,
    ) -> None:
        """Constructor.

        Args:
            root_dir (str): Root directory.
            k_folds (int): Number of folds.
            lightning_module (type[L.LightningModule]): Lightning module.
            lightning_trainer (type[L.Trainer]): Lightning trainer.
            loggers (Sequence[type[L_loggers.Logger]]): Loggers.
            train_patients_txt (str): Paths to the train patients txt file.
            val_patients_txt (str): Paths to the validation patients txt file.
            train_dataset (type[data.Dataset]): Train dataset.
            train_loader (type[data.DataLoader]): Train dataloader.
            val_dataset (type[data.Dataset]): Validation dataset. 
            val_loader (type[data.DataLoader]): Validation dataloader.
            ckpt_path (str | None): Checkpoint path. Default to None.
            seed (int): Seed. Default to 42.
        """
        if len(val_patients_txts) != k_folds:
            raise ValueError(
                "Number of validation patients txts must be equal to k_folds.")
        if len(train_patients_txts) != k_folds:
            raise ValueError(
                "Number of training patients txts must be equal to k_folds.")
        self.k = k_folds
        self.root_dir = root_dir
        self.dirs = [
            os.path.join(root_dir, f"fold_{i}") for i in range(k_folds)]
        self.lightning_module = lightning_module
        self.lighting_trainers = [
            lightning_trainer(
                default_root_dir=d,
                logger=[l(save_dir=d) for l in loggers],
            )
            for d in self.dirs]
        self.train_loaders = [
            train_loader(dataset=train_dataset(patients_txt=t))
            for t in train_patients_txts
        ]
        self.val_loaders = [
            val_loader(dataset=val_dataset(patients_txt=t))
            for t in val_patients_txts
        ]
        self.ckpt_path = ckpt_path
        self.seed = seed

    def fit(self) -> None:
        """Fit the model."""
        L.seed_everything(self.seed)
        val_df = pd.DataFrame(columns=["fold", "val_loss"])
        for i in range(self.k):
            print(f"Training fold {i}.")
            self.lighting_trainers[i].fit(
                model=self.lightning_module(),
                train_dataloaders=self.train_loaders[i],
                val_dataloaders=self.val_loaders[i],
                ckpt_path=self.ckpt_path,
            )
            val_res = self.lighting_trainers[i].validate(
                model=self.lightning_module(),
                dataloaders=self.val_loaders[i],
                ckpt_path="best",
            )
            val_df = pd.concat((
                val_df,
                pd.DataFrame({
                    "fold": [i],
                    "val_loss": [val_res[0]],
                })),
                ignore_index=True,
            )
            val_df.to_csv(os.path.join(self.root_dir, "val_results.csv"))
