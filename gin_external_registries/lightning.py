"""Torchvision module configuration for gin."""

import gin
import lightning as L
import lightning.pytorch.callbacks as callbacks
import lightning.pytorch.loggers as loggers
import lightning.pytorch.plugins as plugins

# Trainer
gin.external_configurable(L.Trainer, module="lightning")

# Loggers
gin.external_configurable(loggers.CSVLogger, module="lightning")
gin.external_configurable(loggers.TensorBoardLogger, module="lightning")

# Callbacks
gin.external_configurable(callbacks.EarlyStopping, module="lightning")
gin.external_configurable(
    callbacks.GradientAccumulationScheduler, module="lightning")
gin.external_configurable(callbacks.ModelCheckpoint, module="lightning")
gin.external_configurable(callbacks.ProgressBar, module="lightning")
gin.external_configurable(callbacks.TQDMProgressBar, module="lightning")
gin.external_configurable(callbacks.RichModelSummary, module="lightning")
gin.external_configurable(callbacks.RichProgressBar, module="lightning")

# Plugins
gin.external_configurable(plugins.DeepSpeedPrecisionPlugin, module="lightning")
gin.external_configurable(plugins.DoublePrecisionPlugin, module="lightning")
gin.external_configurable(plugins.FSDPMixedPrecisionPlugin, module="lightning")
gin.external_configurable(plugins.FSDPPrecisionPlugin, module="lightning")
gin.external_configurable(plugins.HalfPrecisionPlugin, module="lightning")
gin.external_configurable(plugins.MixedPrecisionPlugin, module="lightning")
