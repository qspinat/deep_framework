import os
from typing import Sequence

import gin
from kornia import augmentation
import numpy as np
import torch
from torch import nn
import torchvision

from . import base_dataset as bd
from . import csv_dataset as cd

LABEL_ENCODING = {
    "CC": 0,
    "EC": 1,
    "HGSC": 2,
    "LGSC": 3,
    "MC": 4,
}


@gin.register(module="dataset")
class ImageDataset(bd.BaseDataset):
    """Dataset for images.

    Attributes:
        db_path (str): Path to the database.
        csv_dataset (cd.CSVDataset): CSV dataset.
        images_folder (str): Folder containing the images.
        suffixe (str): Suffixe of the files.
        uids_txt (str | None): Path to a txt file containing the uids.
        preprocess (Sequence[augmentation.AugmentationBase2D] | None): Preprocessing transforms.
        data_aug (Sequence[augmentation.AugmentationBase2D] | None): Data augmentations.
        rescale (tuple[float, float]): Rescale values.
    """

    def __init__(
        self,
        db_path: str,
        csv_dataset: cd.CSVDataset,
        images_folder: str = "train_thumbnails",
        suffixe: str = ".png",
        uids_txt: str | None = None,
        preprocess: Sequence[augmentation.AugmentationBase2D] | None = None,
        data_aug: Sequence[augmentation.AugmentationBase2D] | None = None,
        rescale: tuple[float, float] | None = (0, 255)
    ) -> None:
        """Constructor.

        Args:
            db_path (str): Path to the database.
            csv_dataset (cd.CSVDataset): CSV dataset.
            images_folder (str): Folder containing the images.
                Default to "train_thumbnails".
            suffixe (str): Suffixe of the files. Default to ".png".
            uids_txt (str | None): Path to a txt file containing the uids.
                Default to None.
            preprocess (Sequence[augmentation.AugmentationBase2D] | None):
                Preprocessing transforms. Default to None.
            data_aug (Sequence[augmentation.AugmentationBase2D] | None):
                Data augmentations. Default to None.
            rescale (tuple[float, float] | None): Rescale values.
                Default to (0, 255).
        """
        super().__init__(db_path=db_path,
                         preprocess=preprocess,
                         data_aug=data_aug)
        self.preprocess = nn.Sequential(*self.preprocess)
        self.data_aug = nn.Sequential(*self.data_aug)
        self.images_folder = os.path.join(db_path, images_folder)
        self.csv_dataset = csv_dataset
        self.suffixe = suffixe
        self.rescale = rescale

        self._uids = set(self.csv_dataset.uids)
        self._uids = self._uids & set([uid.split(suffixe)[0]
                                       for uid in os.listdir(self.images_folder)])
        if uids_txt is not None:
            self._uids = (self._uids &
                          set(np.loadtxt(os.path.join(db_path, uids_txt), dtype=str).tolist()))
        self._uids = sorted(list(self._uids))

    def _load_image(self, uid: str) -> torch.Tensor:
        """Load an image.

        Args:
            uid (str): Unique identifier.

        Returns:
            torch.Tensor: Image.
        """
        img_path = os.path.join(
            self.images_folder, f"{uid}{self.suffixe}")
        x = torchvision.io.read_image(path=img_path)
        x = x.float()
        if self.rescale is not None:
            x = (x-self.rescale[0])/(self.rescale[1]-self.rescale[0])
        return x

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        """Get item from dataset.

        Args:
            index (int): Index of the item.

        Returns:
            dict[str, torch.Tensor]: Data. dictwith keys:
                - "input": Input image.
                - "target": Target label.
        """
        uid = self.uids[index]
        # label
        labels = self.csv_dataset.df.loc[
            self.csv_dataset.df.index == uid,
            self.csv_dataset.target_features].values[0].tolist()
        labels = [LABEL_ENCODING[l] for l in labels]
        data = {"target": torch.tensor(labels, dtype=int)}
        # image
        x = self._load_image(uid)
        x = self.preprocess(x)
        if self.is_train():
            x = self.data_aug(x)
        data["input"] = x
        return data
