"""Partial AUROC metric."""

from typing import Any

import gin
import torch
from torchmetrics import classification
from torchmetrics.functional.classification.roc import _binary_roc_compute
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.compute import _auc_compute_without_check


def _binary_auroc_compute_v2(
    state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    thresholds: torch.Tensor | None,
    max_fpr: float | None = None,
    min_tpr: float | None = None,
    pos_label: int = 1,
) -> torch.Tensor:
    if max_fpr and min_tpr:
        raise ValueError(
            "Only one of `max_fpr` and `min_tpr` should be set, not both."
        )
    fpr, tpr, _ = _binary_roc_compute(
        state, thresholds, pos_label)
    if ((max_fpr is None or max_fpr == 1) and
        (min_tpr is None or min_tpr == 0) or
            fpr.sum() == 0 or tpr.sum() == 0):
        return _auc_compute_without_check(fpr, tpr, 1.0)

    _device = fpr.device if isinstance(fpr, torch.Tensor) else fpr[0].device
    if max_fpr is not None:
        max_area: torch.Tensor = torch.tensor(max_fpr).to(_device)
        # Add a single point at max_fpr and interpolate its tpr value
        stop = torch.bucketize(max_area, fpr, out_int32=True, right=True)
        weight = (max_area - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1])
        interp_tpr: torch.Tensor = torch.lerp(tpr[stop - 1], tpr[stop], weight)
        tpr = torch.cat([tpr[:stop], interp_tpr.view(1)])
        fpr = torch.cat([fpr[:stop], max_area.view(1)])
        partial_auc = _auc_compute_without_check(fpr, tpr, 1.0)
    elif min_tpr is not None:
        min_area: torch.Tensor = torch.tensor(min_tpr).to(_device)
        # Add a single point at min_tpr and interpolate its tpr value
        stop = torch.bucketize(min_area, tpr, out_int32=True, right=True)
        weight = (min_area - tpr[stop - 1]) / (tpr[stop] - tpr[stop - 1])
        interp_tpr: torch.Tensor = torch.lerp(fpr[stop - 1], fpr[stop], weight)
        tpr = torch.cat([min_area.view(1), tpr[stop:]])
        fpr = torch.cat([interp_tpr.view(1), fpr[stop:]])
        partial_auc = (
            _auc_compute_without_check(fpr, tpr, 1.0) - tpr[0]*(1-fpr[0]))

    return partial_auc


@gin.register(module="torchmetrics")
class PartialBinaryAUROC(classification.BinaryAUROC):
    """ Partial Binary AUROC metric. Makes it possible possible to use a 
    min_tpr, which torchmetrics doesn't allow."""

    def __init__(self, min_tpr: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_tpr = min_tpr

    def compute(self) -> torch.torch.Tensor:  # type: ignore[override]
        """Compute metric."""
        state = ((dim_zero_cat(self.preds), dim_zero_cat(self.target))
                 if self.thresholds is None else self.confmat)
        return _binary_auroc_compute_v2(
            state, self.thresholds, self.max_fpr, self.min_tpr)
