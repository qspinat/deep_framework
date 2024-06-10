"""This module contains the NiiDataset class, which is a subclass of BaseDataset."""

import concurrent.futures
import os
from typing import Sequence

import gin
import numpy as np
import torch
import torchio as tio

from . import base_dataset as bd
from . import csv_dataset as cd


@gin.register(module="dataset")
class NiiDataset(bd.BaseDataset):
    """Dataset for nii files.

    Attributes:
        db_path (str): Path to the database.
        input_vol_keys (list[str]): Input volume keys.
        target_vol_keys (list[str]): Target volume keys.
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
        label_keys: list[str],
        csv_dataset: cd.CSVDataset | None = None,
        patients_txt: str | None = None,
        preprocess: Sequence[tio.Transform] | None = None,
        data_aug: Sequence[tio.Transform] | None = None,
    ) -> None:
        """Constructor.

        Args:
            db_path (str): Path to the database.
            input_vol_keys (list[str]): Input volume keys.
            target_vol_keys (list[str]): Target volume keys.
            csv_dataset (cd.CSVDataset | None): CSV dataset. Default to None.
            patients_txt (str | None): Path to a txt file containing the patients.
                Default to None.
            preprocess (Sequence[tio.Transform] | None): Preprocessing transforms.
                Default to None.
            data_aug (Sequence[tio.Transform] | None): Data augmentations.
                Default to None.
        """
        super().__init__(db_path=db_path,
                         preprocess=preprocess,
                         data_aug=data_aug)
        self.preprocess = tio.Compose(self.preprocess)
        self.data_aug = tio.Compose(self.data_aug)
        self.input_vol_keys = input_vol_keys
        self.target_vol_keys = target_vol_keys
        self.label_keys = label_keys
        self.csv_dataset = csv_dataset

        self._uids = None
        for key in input_vol_keys + target_vol_keys:
            _uids = os.listdir(os.path.join(db_path, key))
            _uids = [uid.split(".nii")[0] for uid in _uids]
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

    def _load_subject(self, uid: str) -> tio.Subject:
        """Load a subject from its uid.

        Args:
            uid (str): Unique identifier of the subject to load.

        Returns:
            tio.Subject: Subject containing input ant target images.
        """
        subject = {}

        def read_vol(key: str):
            nii_path = os.path.join(
                self.db_path, key, f"{uid}.nii")
            if key in self.label_keys:
                return tio.LabelMap(nii_path)
            return tio.ScalarImage(nii_path)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.input_vol_keys)) as executor:
            vols = executor.map(read_vol, self.input_vol_keys)
        vols = list(vols)
        for vol, key in zip(vols, self.input_vol_keys):
            subject[key] = vol

        def read_mask(key: str):
            nii_path = os.path.join(
                self.db_path, key, f"{uid}.nii.gz")
            if key in self.label_keys:
                return tio.LabelMap(nii_path)
            return tio.ScalarImage(nii_path)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.target_vol_keys)) as executor:
            masks = executor.map(read_mask, self.target_vol_keys)
        masks = list(masks)
        for mask, key in zip(masks, self.target_vol_keys):
            subject[key] = mask
        return tio.Subject(subject)

    def _compute_input(self, subject: tio.Subject) -> torch.Tensor:
        """Compute input tensor from subject. Concatenates input volumes 
        along channel axis.

        Args:
            subject (tio.Subject): Subject containing input and target volumes.

        Returns:
            torch.Tensor: Input tensor.
        """
        return torch.cat(
            [subject[key].data.float() for key in self.input_vol_keys], dim=0)

    def _compute_target(self, subject: tio.Subject) -> torch.Tensor:
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
        subject = self.preprocess(subject)
        if self.is_train():
            subject = self.data_aug(subject)
        data["input"] = self._compute_input(subject)
        data["target"] = self._compute_target(subject)
        return data
