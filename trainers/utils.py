"""Utils for trainers."""

import numpy as np
from torch.utils import data

from deep_framework.datasets import base_dataset


def get_weighted_sampler(
    dataset: base_dataset.BaseDataset,
    weight_per_class: list[float] | None = None,
) -> data.WeightedRandomSampler:
    """Get a weighted sampler for the dataset, which balance target classes in 
    the sampling.

    Args:
        dataset (base_dataset.BaseDataset): Dataset.
        weight_per_class (list[float] | None): Weights per class. If None, the
            weights are calculated as 1/number of samples per class. Defaults
            to None.

    Returns:
        data.WeightedRandomSampler: Weighted sampler.
    """
    labels = dataset.csv_dataset.df.loc[
        dataset.uids, dataset.csv_dataset.target_features[0]
    ].values
    lab, counts = np.unique(labels, return_counts=True)
    if weight_per_class is not None and len(weight_per_class) != len(lab):
        raise ValueError("The number of weights should be equal to the number"
                         "of unique labels.")
    weights = np.ones_like(labels)
    for l, c in zip(lab, counts):
        w = (weight_per_class[int(l)]
             if weight_per_class is not None else 1/c)
        weights[labels == l] = w
    sampler = data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True,
    )
    return sampler
