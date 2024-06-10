"""Base class for datasets."""

import abc
from typing import Literal, Sequence

from kornia import augmentation
from torch.utils import data
import torchio as tio


class BaseDataset(data.Dataset, abc.ABC):
    """Base class for datasets.

    Attributes:
        db_path (str): Path to the database.
        preprocess (Sequence[augmentation.AugmentationBase2D] |
            Sequence[augmentation.AugmentationBase3D]): Preprocessing
            augmentations.
        data_aug (Sequence[augmentation.AugmentationBase2D] |
            Sequence[augmentation.AugmentationBase3D]): Data augmentations.
        mode (Literal["eval", "test", "train"]): Mode of the dataset.

    Properties:
        uids (list): List of unique identifiers.

    Methods:
        eval: Set mode to "eval".
        is_eval: Check if mode is "eval".
        train: Set mode to "train".
        is_train: Check if mode is "train".
        test: Set mode to "test".
        is_test: Check if mode is "test".
    """

    def __init__(
        self,
        db_path: str,
        preprocess: (
            Sequence[augmentation.AugmentationBase2D] |
            Sequence[augmentation.AugmentationBase3D] |
            Sequence[tio.Transform] | None
        ) = None,
        data_aug: (
            Sequence[augmentation.AugmentationBase2D] |
            Sequence[augmentation.AugmentationBase3D] |
            Sequence[tio.Transform] | None
        ) = None,
    ) -> None:
        """ Constructor.

        Args:
            db_path (str): Path to the database.
            preprocess (Sequence[augmentation.AugmentationBase2D] |
                Sequence[augmentation.AugmentationBase3D] |
                tio.Transform | None): Preprocessing augmentations.
            data_aug (Sequence[augmentation.AugmentationBase2D] |
                Sequence[augmentation.AugmentationBase3D] | 
                tio.Transforms | None): Data augmentations.
        """
        super().__init__()
        self.db_path = db_path
        preprocess = [] if preprocess is None else preprocess
        data_aug = [] if data_aug is None else data_aug
        self.preprocess = preprocess
        self.data_aug = data_aug
        self.mode: Literal["eval", "test", "train"] = "train"
        self._uids: list

    def eval(self):
        self.mode = "eval"

    def is_eval(self) -> bool:
        return self.mode == "eval"

    def train(self):
        self.mode = "train"

    def is_train(self) -> bool:
        return self.mode == "train"

    def test(self):
        self.mode = "test"

    def is_test(self) -> bool:
        return self.mode == "test"

    def __len__(self) -> int:
        return len(self._uids)

    @property
    def uids(self) -> list:
        return self._uids

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Get item from dataset. 

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Dictionary with the data.
        """
