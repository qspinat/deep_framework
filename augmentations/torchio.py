""" torchio augmentations. """

from typing import Sequence

import gin
import numpy as np
import torch
import torchio as tio

from skimage import morphology


@gin.register(module="tio")
class Dilatation(tio.Transform):
    """Dilatation transform for binary LabelMaps."""

    def __init__(self, dilatation: int, **kwargs) -> None:
        """Constructor.

        Args:
            dilation (int): Dilation factor.
        """
        super().__init__(**kwargs)
        self.dilatation = dilatation

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Resample apply dilatation on volume."""
        im_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for key, image in im_dict.items():
            if not isinstance(image, tio.LabelMap):
                continue

            dilated_label = image.data.clone()
            for i in range(dilated_label.shape[0]):
                dilated_label[i] = torch.from_numpy(morphology.binary_dilation(
                    image.data[i].numpy().astype(bool),
                    morphology.ball(self.dilatation)
                ).astype(int)).to(dilated_label)

            image.set_data(
                dilated_label
            )

            subject[key] = image
        return subject


@gin.register(module="tio")
class RandomCrop(
        tio.transforms.augmentation.RandomTransform,
        tio.transforms.SpatialTransform):
    """Randomly crop a region from the input subject."""

    def __init__(self, *args, crop_shape: Sequence[int], **kwargs):
        """Randomly crop a region from the input subject.

        This class is a TorchIO transform that performs random cropping on a
        3D image from the input subject. The cropping region is randomly
        selected based on the specified crop shape. The transform is typically
        used for data augmentation in the context of deep learning.

        Args:
            crop_shape (Sequence[int]): Dimensions of the cropping region in
                [depth, height, width].

        Attributes:
            crop_shape (Sequence[int]): Dimensions of the cropping region in
                [depth, height, width].

        Note:
            This transform uses the TorchIO `Crop` transform internally to
            perform the actual cropping.
            The notation `ini` and `fin` are from the doc of torchio
            https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.Crop
        """
        super().__init__(*args, **kwargs)
        self.crop_shape = crop_shape

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        image = subject.get_images(
            include=self.include, exclude=self.exclude)[0]
        min_ini = np.zeros_like(image.data.shape[1:])
        max_ini = np.asarray(
            image.data.shape[1:]) - np.array(self.crop_shape) + 1
        if max_ini.min() <= 0:
            max_ini = np.maximum(max_ini, np.ones_like(max_ini))
        ini = np.random.randint(min_ini, max_ini).astype(int)
        fin = (
            np.array(image.data.shape[1:]) - (ini + np.array(self.crop_shape))
        ).astype(int)
        fin = np.maximum(fin, np.zeros_like(fin))
        bound = (ini[0], fin[0], ini[1], fin[1], ini[2], fin[2])
        crop = tio.Crop(
            cropping=bound,
            include=self.include,
            exclude=self.exclude,
            keep=self.keep,
            copy=self.copy,
            parse_input=self.parse_input,
            label_keys=self.label_keys,
        )
        return crop(subject)


@gin.register(module="tio")
class RandomDropout(
        tio.transforms.augmentation.RandomTransform,
        tio.transforms.SpatialTransform):
    """ Randomly dropout. """

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        im_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for key, image in im_dict.items():
            image.set_data(torch.zeros_like(image.data))
            subject[key] = image
        return subject
