import os
import concurrent.futures
import random
from typing import Sequence

import gin
from kornia import augmentation
import torch
import torchvision.io

from deep_framework.datasets import csv_dataset as cd

from .import csv_dataset as cd
from .import image_dataset as id


LABEL_ENCODING = {
    "CC": 0,
    "EC": 1,
    "HGSC": 2,
    "LGSC": 3,
    "MC": 4,
}


@gin.register(module="datasets")
class PatchDataset(id.ImageDataset):
    def __init__(self,
                 db_path: str,
                 csv_dataset: cd.CSVDataset,
                 images_folder: str = "train_images",
                 suffixe: str = ".png", uids_txt: str | None = None,
                 preprocess: Sequence[augmentation.AugmentationBase2D] | None = None,
                 data_aug: Sequence[augmentation.AugmentationBase2D] | None = None,
                 rescale: tuple[float, float] | None = (0, 255),
                 n_max_patches: int = -1,
                 n_threads: int = 8,
                 ) -> None:
        super().__init__(
            db_path=db_path,
            csv_dataset=csv_dataset,
            images_folder=images_folder,
            suffixe=suffixe,
            uids_txt=uids_txt,
            preprocess=preprocess,
            data_aug=data_aug,
            rescale=rescale
        )
        self.n_max_patches = n_max_patches
        self.n_threads = n_threads

    def _load_image(self, uid: str) -> torch.Tensor:
        patch_folder = os.path.join(
            self.images_folder, uid)
        patch_files = os.listdir(patch_folder)
        if self.n_max_patches > 0 and len(patch_files) > self.n_max_patches:
            patch_files = random.sample(patch_files, self.n_max_patches)
        patch_files = [os.path.join(patch_folder, p) for p in patch_files]
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.n_threads) as executor:
            patches = executor.map(torchvision.io.read_image, patch_files)
        patches = list(patches)
        patches = torch.stack(patches, 0)
        patches = patches.float()
        if self.rescale is not None:
            patches = (
                patches-self.rescale[0])/(self.rescale[1]-self.rescale[0])
        return patches


@gin.register(module="datasets")
class PatchDatasetWholeImage(id.ImageDataset):
    def __init__(self,
                 db_path: str,
                 csv_dataset: cd.CSVDataset,
                 images_folder: str = "train_images",
                 suffixe: str = ".png", uids_txt: str | None = None,
                 preprocess: Sequence[augmentation.AugmentationBase2D] | None = None,
                 data_aug: Sequence[augmentation.AugmentationBase2D] | None = None,
                 rescale: tuple[float, float] | None = (0, 255),
                 n_max_patches: int = -1,
                 n_threads: int = 8,
                 ) -> None:
        super().__init__(
            db_path=db_path,
            csv_dataset=csv_dataset,
            images_folder=images_folder,
            suffixe=suffixe,
            uids_txt=uids_txt,
            preprocess=preprocess,
            data_aug=data_aug,
            rescale=rescale
        )
        self.n_max_patches = n_max_patches
        self.n_threads = n_threads

        # order batches to have he same wsi.
        self.uid_to_patches = {}
        for uid in self._uids:
            patch_folder = os.path.join(
                self.images_folder, uid)
            self.uid_to_patches[uid] = [
                os.path.join(db_path, images_folder, uid, p)
                for p in os.listdir(patch_folder)
            ]
            random.shuffle(self.uid_to_patches[uid])
        # random.shuffle(self._uids)
        self.indice_to_patches = []
        self.indice_to_uid = []
        self.indice_to_label = []
        for i in range(len(self._uids)):
            uid = self._uids[i]
            labels = self.csv_dataset.df.loc[
                self.csv_dataset.df.index == uid,
                self.csv_dataset.target_features].values[0].tolist()
            labels = [LABEL_ENCODING[l] for l in labels]
            patches = self.uid_to_patches[uid]
            n_batches = len(patches)//n_max_patches
            for j in range(n_batches):
                patch_uids = patches[j*n_max_patches:(j+1)*n_max_patches]
                self.indice_to_patches.append(patch_uids)
                self.indice_to_uid.append(uid)
                self.indice_to_label.append(labels)
            if len(patches) % n_max_patches != 0:
                patch_uids = patches[n_batches*n_max_patches:]
                self.indice_to_patches.append(patch_uids)
                self.indice_to_uid.append(uid)
                self.indice_to_label.append(labels)

        # precompute labels

    def __len__(self) -> int:
        return len(self.indice_to_patches)

    def _load_image(self, patch_files: Sequence[str]) -> torch.Tensor:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.n_threads) as executor:
            patches = executor.map(torchvision.io.read_image, patch_files)
        patches = list(patches)
        patches = torch.stack(patches, 0)
        patches = patches.float()
        if self.rescale is not None:
            patches = (
                patches-self.rescale[0])/(self.rescale[1]-self.rescale[0])
        return patches

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        # label
        labels = self.indice_to_label[index]
        data = {"target": torch.tensor(labels, dtype=int)}
        # image
        patch_files = self.indice_to_patches[index]
        x = self._load_image(patch_files)
        x = self.preprocess(x)
        if self.is_train():
            x = self.data_aug(x)
        data["input"] = x
        return data
