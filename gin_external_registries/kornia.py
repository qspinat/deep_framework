"""Kornia module configuration forgin."""

import gin
import kornia

gin.external_configurable(kornia.augmentation.RandomAffine,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomAffine3D,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomDepthicalFlip3D,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomGamma,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomGaussianBlur,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomGaussianNoise,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomHorizontalFlip,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomHorizontalFlip3D,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomMotionBlur,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomMotionBlur3D,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomRotation,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomRotation3D,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomVerticalFlip,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomVerticalFlip3D,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomContrast,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomPlanckianJitter,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomPlasmaShadow,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomPlasmaBrightness,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomPlasmaShadow,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomPlasmaContrast,
                          module="kornia")

gin.external_configurable(kornia.augmentation.ColorJiggle,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomBoxBlur,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomBrightness,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomChannelShuffle,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomEqualize,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomEqualize3D,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomGrayscale,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomHue,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomPosterize,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomRGBShift,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomSaturation,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomSharpness,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomSolarize,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomMedianBlur,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomSnow,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomRain,
                          module="kornia")

gin.external_configurable(kornia.augmentation.CenterCrop,
                          module="kornia")

gin.external_configurable(kornia.augmentation.CenterCrop3D,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomSnow,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomRain,
                          module="kornia")

gin.external_configurable(kornia.augmentation.PadTo,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomCrop,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomCrop3D,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomErasing,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomElasticTransform,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomInvert,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomPerspective,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomPerspective3D,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomResizedCrop,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomThinPlateSpline,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomMosaic,
                          module="kornia")

gin.external_configurable(kornia.augmentation.RandomJigsaw,
                          module="kornia")

gin.external_configurable(kornia.augmentation.Denormalize,
                          module="kornia")

gin.external_configurable(kornia.augmentation.Normalize,
                          module="kornia")

gin.external_configurable(kornia.augmentation.Resize,
                          module="kornia")

gin.external_configurable(kornia.augmentation.LongestMaxSize,
                          module="kornia")

gin.external_configurable(kornia.augmentation.SmallestMaxSize,
                          module="kornia")
