"""Utils for trainers."""

import numpy as np
from torch.utils import data

from deep_framework.datasets import base_dataset


def get_weighted_sampler(
    dataset: base_dataset.BaseDataset
) -> data.WeightedRandomSampler:
    """Get a weighted sampler for the dataset, which balance target classes in 
    the sampling.

    Args:
        dataset (base_dataset.BaseDataset): Dataset.

    Returns:
        data.WeightedRandomSampler: Weighted sampler.
    """
    labels = dataset.csv_dataset.df.loc[
        dataset.uids, dataset.csv_dataset.target_features[0]
    ].values
    lab, counts = np.unique(labels, return_counts=True)
    weights = np.ones_like(labels)
    for l, c in zip(lab, counts):
        weights[labels == l] = 1/c
    sampler = data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True,
    )
    return sampler
