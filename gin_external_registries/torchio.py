import gin
from torchio import transforms

# Transform selection
gin.external_configurable(transforms.Compose, module="tio")
gin.external_configurable(transforms.OneOf, module="tio")

# Data augmentations
gin.external_configurable(transforms.RandomAffine, module="tio")
gin.external_configurable(transforms.RandomAnisotropy, module="tio")
gin.external_configurable(transforms.RandomBiasField, module="tio")
gin.external_configurable(transforms.RandomBlur, module="tio")
gin.external_configurable(transforms.RandomElasticDeformation, module="tio")
gin.external_configurable(transforms.RandomFlip, module="tio")
gin.external_configurable(transforms.RandomGamma, module="tio")
gin.external_configurable(transforms.RandomGhosting, module="tio")
gin.external_configurable(transforms.RandomLabelsToImage, module="tio")
gin.external_configurable(transforms.RandomMotion, module="tio")
gin.external_configurable(transforms.RandomNoise, module="tio")
gin.external_configurable(transforms.RandomSpike, module="tio")
gin.external_configurable(transforms.RandomSwap, module="tio")

# Pre-processing
gin.external_configurable(transforms.Clamp, module="tio")
gin.external_configurable(transforms.CropOrPad, module="tio")
gin.external_configurable(transforms.RemapLabels, module="tio")
gin.external_configurable(transforms.Resample, module="tio")
gin.external_configurable(transforms.RescaleIntensity, module="tio")
gin.external_configurable(transforms.ZNormalization, module="tio")
