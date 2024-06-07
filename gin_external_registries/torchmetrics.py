"""Torchmetric module configuration for gin."""

import gin
import torchmetrics

gin.external_configurable(torchmetrics.Accuracy, module="torchmetrics")
gin.external_configurable(torchmetrics.AUROC, module="torchmetrics")
gin.external_configurable(torchmetrics.AveragePrecision, module="torchmetrics")
gin.external_configurable(torchmetrics.CalibrationError, module="torchmetrics")
gin.external_configurable(torchmetrics.F1Score, module="torchmetrics")
gin.external_configurable(torchmetrics.Precision, module="torchmetrics")
gin.external_configurable(
    torchmetrics.MeanAbsoluteError, module="torchmetrics")
gin.external_configurable(torchmetrics.MeanSquaredError, module="torchmetrics")
gin.external_configurable(torchmetrics.Recall, module="torchmetrics")
gin.external_configurable(torchmetrics.R2Score, module="torchmetrics")
gin.external_configurable(torchmetrics.Specificity, module="torchmetrics")
gin.external_configurable(
    torchmetrics.StructuralSimilarityIndexMeasure, module="torchmetrics")
