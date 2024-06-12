""" torchio augmentations. """

import gin
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
