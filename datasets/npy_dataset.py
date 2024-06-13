"""This module contains the NiiDataset class, which is a subclass of BaseDataset."""

import concurrent.futures
import os
import time
from typing import Sequence

import gin
import kornia
import numpy as np
import torch
from torch import nn

from . import base_dataset as bd
from . import csv_dataset as cd


@gin.register(module="dataset")
class NpyDataset(bd.BaseDataset):
    """Dataset for npy files.

    Attributes:
        db_path (str): Path to the database.
        input_vol_keys (list[str]): Input volume keys.
        target_vol_keys (list[str]): Target volume keys.
        compressed_keys (list[str]): Keys to load as npz.
        csv_dataset (cd.CSVDataset | None): CSV dataset.
        patients_txt (str | None): Path to a txt file containing the patients.
        preprocess (Sequence[tio.Transform] | None): Preprocessing transforms.
        data_aug (Sequence[tio.Transform] | None): Data augmentations.

    Properties:
        uids (list): List of unique identifiers.
    """

    def __init__(
        self,
        db_path: str,
        input_vol_keys: list[str],
        target_vol_keys: list[str],
        compressed_keys: list[str] | None = None,
        csv_dataset: cd.CSVDataset | None = None,
        patients_txt: str | None = None,
        preprocess: (Sequence[kornia.augmentation.AugmentationBase2D |
                     kornia.augmentation.AugmentationBase2D] | None) = None,
        data_aug: (Sequence[kornia.augmentation.AugmentationBase2D |
                   kornia.augmentation.AugmentationBase2D] | None) = None,
    ) -> None:
        """Constructor.

        Args:
            db_path (str): Path to the database.
            input_vol_keys (list[str]): Input volume keys.
            target_vol_keys (list[str]): Target volume keys.
            compressed_keys (list[str]): Keys to load as npz. Default to None.
            csv_dataset (cd.CSVDataset | None): CSV dataset. Default to None.
            patients_txt (str | None): Path to a txt file containing the patients.
                Default to None.
            preprocess (Sequence[kornia.augmentation.AugmentationBase2D |
                kornia.augmentation.AugmentationBase2D] | None): Preprocessing 
                transforms. Default to None.
            data_aug (Sequence[kornia.augmentation.AugmentationBase2D |
                kornia.augmentation.AugmentationBase2D] | None): Data 
                augmentations. Default to None.
        """
        super().__init__(
            db_path=db_path,
            preprocess=preprocess,
            data_aug=data_aug)
        self.preprocess = nn.Sequential(*self.preprocess)
        self.data_aug = nn.Sequential(*self.data_aug)
        self.input_vol_keys = input_vol_keys
        self.target_vol_keys = target_vol_keys
        self.compressed_keys = (
            compressed_keys if compressed_keys is not None else [])
        self.csv_dataset = csv_dataset

        self._uids = None
        for key in input_vol_keys + target_vol_keys:
            _uids = os.listdir(os.path.join(db_path, key))
            _uids = [uid.split(".np")[0] for uid in _uids]
            self._uids = (set(_uids) if self._uids is None
                          else self._uids & set(_uids))

        if csv_dataset is not None:
            self._uids = self._uids & set(self.csv_dataset.uids)

        self._uids = list(self._uids)

        if patients_txt is not None:
            patients = np.loadtxt(os.path.join(
                db_path, patients_txt), dtype=str).tolist()
            self._uids = [
                uid for uid in self._uids if uid.split("_")[0] in patients
            ]
        self._uids = sorted(self._uids)

    def _load_subject(self, uid: str) -> dict[str, torch.Tensor]:
        """Load a subject from its uid.

        Args:
            uid (str): Unique identifier of the subject to load.

        Returns:
            tio.Subject: Subject containing input ant target images.
        """
        subject = {}

        def read_vol(key: str):
            if key in self.compressed_keys:
                npy_path = os.path.join(
                    self.db_path, key, f"{uid}.npz")
            else:
                npy_path = os.path.join(
                    self.db_path, key, f"{uid}.npy")
            attempt = 0
            while attempt < 15:
                try:
                    vol = np.load(npy_path)
                    break
                except (ValueError, EOFError, FileNotFoundError) as e:
                    print(e)
                    attempt += 1
                    time.sleep(1e-1)
            return torch.from_numpy(vol.astype(np.float32))
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.input_vol_keys)) as executor:
            vols = executor.map(read_vol, self.input_vol_keys)
        vols = list(vols)
        for vol, key in zip(vols, self.input_vol_keys):
            subject[key] = vol
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.target_vol_keys)) as executor:
            masks = executor.map(read_vol, self.target_vol_keys)
        masks = list(masks)
        for mask, key in zip(masks, self.target_vol_keys):
            subject[key] = mask
        return subject

    def _compute_input(self, subject: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute input tensor from subject. Concatenates input volumes 
        along channel axis.

        Args:
            subject (tio.Subject): Subject containing input and target volumes.

        Returns:
            torch.Tensor: Input tensor.
        """
        return torch.cat(
            [subject[key].data.float() for key in self.input_vol_keys], dim=0)

    def _compute_target(self, subject: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute target tensor from subject. Concatenates target volumes
        along channel axis.

        Args:
            subject (tio.Subject): Subject containing input and target volumes.

        Returns:
            torch.Tensor: Target tensor.
        """
        return torch.cat(
            [subject[key].data.float() for key in self.target_vol_keys], dim=0)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        """Get item from dataset.

        Args:
            index (int): Index of the item.

        Returns:
            dict[str, torch.Tensor]: Data. Contains the keys:
                - "input": Input tensor.
                - "target": Target tensor.
        """
        uid = self.uids[index]
        data = {}
        subject = self._load_subject(uid)
        data["input"] = self._compute_input(subject)
        data["target"] = self._compute_target(subject)
        data["input"] = self.preprocess(data["input"])
        data["target"] = self.preprocess(data["target"])
        if self.is_train():
            data["input"] = self.data_aug(data["input"])
        return data
