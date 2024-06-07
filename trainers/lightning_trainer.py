"""lightning trainer class."""

import gin
from torch.utils import data
import lightning as L


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
            seed (int): Seed. Default to 42.
        """
        self.lightning_module = lightning_module()
        self.lighting_trainer = lightning_trainer()
        self.train_loader = train_loader()
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
