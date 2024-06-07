""" Custom collate function for datasets. """

import gin


@gin.register(module="datasets")
def no_batch_collate(batch):
    return batch[0]
