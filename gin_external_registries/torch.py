"""torch configuration for gin."""

import gin
from gin.torch import external_configurables
import torch
from torch.utils import data

# Dataloader
gin.external_configurable(data.DataLoader, module="torch")

# Tensor creation
gin.external_configurable(torch.arange, module="torch")
gin.external_configurable(torch.full, module="torch")
gin.external_configurable(torch.full_like, module="torch")
gin.external_configurable(torch.linspace, module="torch")
gin.external_configurable(torch.logspace, module="torch")
gin.external_configurable(torch.ones, module="torch")
gin.external_configurable(torch.ones_like, module="torch")
gin.external_configurable(torch.range, module="torch")
gin.external_configurable(torch.tensor, module="torch")
gin.external_configurable(torch.zeros, module="torch")
gin.external_configurable(torch.zeros_like, module="torch")

# Operations
gin.external_configurable(torch.cat, module="torch")
gin.external_configurable(torch.concat, module="torch")
gin.external_configurable(torch.dstack, module="torch")
gin.external_configurable(torch.hstack, module="torch")
gin.external_configurable(torch.squeeze, module="torch")
gin.external_configurable(torch.stack, module="torch")
gin.external_configurable(torch.unsqueeze, module="torch")
gin.external_configurable(torch.vstack, module="torch")

# Generator
gin.external_configurable(torch.Generator, module="torch")

# Random sampling
gin.external_configurable(torch.rand, module="torch")
gin.external_configurable(torch.rand_like, module="torch")
gin.external_configurable(torch.randint, module="torch")
gin.external_configurable(torch.randint_like, module="torch")
gin.external_configurable(torch.randn, module="torch")
gin.external_configurable(torch.randn_like, module="torch")

# Activation functions
gin.external_configurable(torch.sigmoid, module="torch")
gin.external_configurable(torch.softmax, module="torch")
gin.external_configurable(torch.tanh, module="torch")

# Optimizers
gin.external_configurable(torch.optim.NAdam, module="torch")
gin.external_configurable(torch.optim.RAdam, module="torch")

# Schedulers
gin.external_configurable(torch.optim.lr_scheduler.LinearLR, module="torch")

# Convolutional Layers
gin.external_configurable(torch.nn.Conv1d, module="torch")
gin.external_configurable(torch.nn.Conv2d, module="torch")
gin.external_configurable(torch.nn.Conv3d, module="torch")
gin.external_configurable(torch.nn.ConvTranspose1d, module="torch")
gin.external_configurable(torch.nn.ConvTranspose2d, module="torch")
gin.external_configurable(torch.nn.ConvTranspose3d, module="torch")

# Normalization Layers
gin.external_configurable(torch.nn.BatchNorm1d, module="torch")
gin.external_configurable(torch.nn.BatchNorm2d, module="torch")
gin.external_configurable(torch.nn.BatchNorm3d, module="torch")
gin.external_configurable(torch.nn.GroupNorm, module="torch")
gin.external_configurable(torch.nn.InstanceNorm1d, module="torch")
gin.external_configurable(torch.nn.InstanceNorm2d, module="torch")
gin.external_configurable(torch.nn.InstanceNorm3d, module="torch")
gin.external_configurable(torch.nn.LayerNorm, module="torch")

# Dropout
gin.external_configurable(torch.nn.Dropout, module="torch")
gin.external_configurable(torch.nn.Dropout1d, module="torch")
gin.external_configurable(torch.nn.Dropout2d, module="torch")
gin.external_configurable(torch.nn.Dropout3d, module="torch")

# Gradient clipping
gin.external_configurable(torch.nn.utils.clip_grad_norm_, module="torch")
gin.external_configurable(torch.nn.utils.clip_grad_value_, module="torch")

# Activations
gin.external_configurable(torch.nn.GELU, module="torch")
gin.external_configurable(torch.nn.Identity, module="torch")

# Pooling
gin.external_configurable(torch.nn.AdaptiveAvgPool1d, module="torch")
gin.external_configurable(torch.nn.AdaptiveAvgPool2d, module="torch")
gin.external_configurable(torch.nn.AdaptiveAvgPool3d, module="torch")
gin.external_configurable(torch.nn.AdaptiveMaxPool1d, module="torch")
gin.external_configurable(torch.nn.AdaptiveMaxPool2d, module="torch")
gin.external_configurable(torch.nn.AdaptiveMaxPool3d, module="torch")
gin.external_configurable(torch.nn.AvgPool1d, module="torch")
gin.external_configurable(torch.nn.AvgPool2d, module="torch")
gin.external_configurable(torch.nn.AvgPool3d, module="torch")
gin.external_configurable(torch.nn.MaxPool1d, module="torch")
gin.external_configurable(torch.nn.MaxPool2d, module="torch")
gin.external_configurable(torch.nn.MaxPool3d, module="torch")

# Transformers
gin.external_configurable(torch.nn.TransformerEncoder, module="torch")
gin.external_configurable(torch.nn.TransformerEncoderLayer, module="torch")

# Losses
gin.external_configurable(torch.nn.CrossEntropyLoss, module="torch")
gin.external_configurable(torch.nn.MSELoss, module="torch")
gin.external_configurable(torch.nn.L1Loss, module="torch")
gin.external_configurable(torch.nn.BCELoss, module="torch")
gin.external_configurable(torch.nn.BCEWithLogitsLoss, module="torch")
gin.external_configurable(torch.nn.SmoothL1Loss, module="torch")
