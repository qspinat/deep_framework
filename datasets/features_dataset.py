""" Module to load features from a folder of npy files. 
Used for UBC-OCEAN challenge."""

import os
import concurrent.futures
import random
from typing import Sequence

import gin
import numpy as np
import torch

from .import csv_dataset as cd
from .import base_dataset as bd


LABEL_ENCODING = {
    "CC": 0,
    "EC": 1,
    "HGSC": 2,
    "LGSC": 3,
    "MC": 4,
}


@gin.register(module="datasets")
class FeaturesDataset(bd.BaseDataset):
    """ Dataset for features from npy files. Used for UBC-OCEAN challenge.
    It loads pre-computed patches features from a folder of npy files.

    Attributes:
        db_path (str): Path to the database.
        csv_dataset (cd.CSVDataset): CSV dataset.
        features_folder (str): Folder containing the features.
        suffixe (str): Suffixe of the files.
        n_max_tokens (int): Maximum number of tokens per batch.
        n_threads (int): Number of threads to load the features.
        data_aug (bool): Wheter to use data augmentation or not.
        indice_to_tokens (list): Index to token matching.
        indice_to_folder (list): Index to folder matching.
        indice_to_label (list): Index to label matching.
    """

    def __init__(self,
                 db_path: str,
                 csv_dataset: cd.CSVDataset,
                 features_folder: str = "train_images",
                 suffixe: str = ".npy",
                 uids_txt: str | None = None,
                 n_max_tokens: int = -1,
                 n_threads: int = 8,
                 data_aug: bool = False
                 ) -> None:
        """ Constructor.

        Args:
            db_path (str): Path to the database.
            csv_dataset (cd.CSVDataset): CSV dataset.
            features_folder (str): Folder containing the features. 
                Default to "train_images".
            suffixe (str): Suffixe of the files. Default to ".npy".
            uids_txt (str | None): Path to the uids txt file. Default to None.
            n_max_tokens (int): Maximum number of tokens per batch. If -1,
                all tokens are used and there is no maximum. Default to -1.
            n_threads (int): Number of threads to load the features. 
                Default to 8.
            data_aug (bool): Wheter to use data augmentation or not.
                Default to False.
            """
        super().__init__(
            db_path=db_path,
            preprocess=None,
            data_aug=None,
        )
        self.n_max_tokens = n_max_tokens
        self.n_threads = n_threads
        self.suffixe = suffixe

        self.features_folder = os.path.join(db_path, features_folder)
        self.csv_dataset = csv_dataset

        self._uids = set(self.csv_dataset.uids)
        if uids_txt is not None:
            self._uids = (
                self._uids & set(np.loadtxt(
                    os.path.join(db_path, uids_txt), dtype=str).tolist()))
        self.feature_folders = os.listdir(self.features_folder)
        self.feature_folders = [
            f for f in self.feature_folders if f.split("-")[0] in self._uids]
        if not data_aug:
            self.feature_folders = [
                f for f in self.feature_folders if f.split("-")[1] == "0"]
        self._uids = self._uids & set(
            [uid.split("-")[0] for uid in self.feature_folders])
        self._uids = sorted(list(self._uids))

        # order batches to have the same folder.
        self.folder_to_tokens = {}
        folders_to_remove = []
        for f_folder in self.feature_folders:
            tokens_folder = os.path.join(self.features_folder, f_folder)
            self.folder_to_tokens[f_folder] = [
                os.path.join(self.features_folder, f_folder, p)
                for p in os.listdir(tokens_folder)
            ]
            if len(self.folder_to_tokens[f_folder]) == 0:
                folders_to_remove.append(f_folder)
            random.shuffle(self.folder_to_tokens[f_folder])
        for f_folder in folders_to_remove:
            self.feature_folders.remove(f_folder)
        self.indice_to_tokens = []
        self.indice_to_folder = []
        self.indice_to_label = []
        for i, f_folder in enumerate(self.feature_folders):
            uid = f_folder.split("-")[0]
            labels = self.csv_dataset.df.loc[
                self.csv_dataset.df.index == uid,
                self.csv_dataset.target_features].values[0].tolist()
            labels = [LABEL_ENCODING[l] for l in labels]
            tokens = self.folder_to_tokens[f_folder]
            if n_max_tokens > 0:
                n_batches = len(tokens)//n_max_tokens
                for j in range(n_batches):
                    batch_tokens = tokens[j*n_max_tokens:(j+1)*n_max_tokens]
                    self.indice_to_tokens.append(batch_tokens)
                    self.indice_to_folder.append(f_folder)
                    self.indice_to_label.append(labels)
                if len(tokens) % n_max_tokens != 0:
                    batch_tokens = tokens[n_batches*n_max_tokens:]
                    self.indice_to_tokens.append(batch_tokens)
                    self.indice_to_folder.append(uid)
                    self.indice_to_label.append(labels)
            else:
                self.indice_to_tokens.append(tokens)
                self.indice_to_folder.append(f_folder)
                self.indice_to_label.append(labels)

    def __len__(self) -> int:
        return len(self.indice_to_tokens)

    def _load_feature(self, token_files: Sequence[str]) -> torch.Tensor:
        """Load features from a list of npy files.

        Args:
            token_files (Sequence[str]): List of npy files to load.

        Returns:
            torch.Tensor: Features.
        """
        def read_fn(path: str) -> torch.Tensor:
            return torch.from_numpy(np.load(path))
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.n_threads) as executor:
            tokens = executor.map(read_fn, token_files)
        tokens = list(tokens)
        tokens = torch.stack(tokens, 0)
        tokens = tokens.float()
        return tokens

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        """Get item from dataset.

        Args:
            index (int): Index of the item.

        Returns:
            dict[str, torch.Tensor]: Data. Contains the keys:
                - "input": Features.
                - "target": Labels.
        """
        labels = self.indice_to_label[index]
        data = {"target": torch.tensor(labels, dtype=int)}
        token_files = self.indice_to_tokens[index]
        x = self._load_feature(token_files)
        data["input"] = x
        return data
