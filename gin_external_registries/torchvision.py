"""Torchvision module configuration for gin."""

import gin
import torchvision

gin.external_configurable(
    torchvision.transforms.CenterCrop, module="torchvision.transforms")
gin.external_configurable(
    torchvision.transforms.Compose, module="torchvision.transforms")
gin.external_configurable(
    torchvision.transforms.Normalize, module="torchvision.transforms")
gin.external_configurable(
    torchvision.transforms.Resize, module="torchvision.transforms")
gin.external_configurable(
    torchvision.transforms.ToTensor, module="torchvision.transforms")
