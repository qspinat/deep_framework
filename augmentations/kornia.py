""" kornia augmentations."""

from typing import Any, Dict, Optional, Tuple

import gin
import torch

from kornia import constants
from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.intensity import base
from kornia.core import check
from kornia.core import Tensor
from kornia.enhance import adjust
from kornia.filters import filter
from kornia.filters import kernels


@gin.register(module="kornia")
class RandomChannelDropout3D(base.IntensityAugmentationBase3D):
    """dropout the channels of a batch of multi-dimensional images."""

    def __init__(
            self,
            same_on_batch: bool = False,
            p: float = 0.5,
            keepdim: bool = False,
            dropout_p: float = 0.5) -> None:
        super().__init__(
            p=p,
            same_on_batch=same_on_batch,
            p_batch=1.0,
            keepdim=keepdim
        )
        self.dropout_p = dropout_p

    def generate_parameters(self, shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        B, C, _, _, _ = shape
        channels = torch.rand(B, C)
        return {"channels": channels}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        channels = params["channels"].to(input)
        input = input*(channels[..., None, None, None] > self.dropout_p)
        return input

    def compute_transformation(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any]
    ) -> torch.Tensor:
        identity = torch.eye(4, dtype=input.dtype, device=input.device)[None,]
        return identity.expand(input.shape[0], 4, 4)


def filter3d_separable(
    input: torch.Tensor,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    kernel_z: torch.Tensor,
    border_type: str = 'reflect',
    normalized: bool = False,
    padding: str = 'same',
) -> torch.Tensor:
    r"""Convolve a tensor with three 1d kernels, in x, y and z directions.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W, D)`.
        kernel_x: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kW)` or :math:`(B, kW)`.
        kernel_y: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH)` or :math:`(B, kH)`.
        kernel_z: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kD)` or :math:`(B, kD)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized..

    Return:
        Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W, D)`.
    """
    out = filter.filter3d(input, kernel_x[..., None, None, :],
                          border_type, normalized)
    out = filter.filter3d(out, kernel_y[..., None, :, None],
                          border_type, normalized)
    out = filter.filter3d(out, kernel_z[..., :, None, None],
                          border_type, normalized)
    return out


def gaussian_blur3d(
    input: torch.Tensor,
    kernel_size: tuple[int, int, int] | int,
    sigma: tuple[float, float] | torch.Tensor,
    border_type: str = 'reflect',
    separable: bool = True,
) -> torch.Tensor:
    r"""Create an operator that blurs a tensor using a Gaussian filter.

    .. image:: _static/img/gaussian_blur2d.png

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        input: the input tensor with shape :math:`(B,C,H,W,D)`.
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of three 1d-convolutions.

    Returns:
        the blurred tensor with shape :math:`(B, C, H, W, D)`.
    """
    check.KORNIA_CHECK_IS_TENSOR(input)

    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=input.device, dtype=input.dtype)
    else:
        check.KORNIA_CHECK_IS_TENSOR(sigma)
        sigma = sigma.to(device=input.device, dtype=input.dtype)

    if separable:
        kz, ky, kx = kernels._unpack_3d_ks(kernel_size)
        bs = sigma.shape[0]
        kernel_x = kernels.get_gaussian_kernel1d(kx, sigma[:, 2].view(bs, 1))
        kernel_y = kernels.get_gaussian_kernel1d(ky, sigma[:, 1].view(bs, 1))
        kernel_z = kernels.get_gaussian_kernel1d(kz, sigma[:, 0].view(bs, 1))
        out = filter3d_separable(
            input, kernel_x, kernel_y, kernel_z, border_type)
    else:
        kernel = kernels.get_gaussian_kernel3d(kernel_size, sigma)
        out = filter.filter3d(input, kernel, border_type)

    return out


@gin.register(module="kornia")
class RandomGaussianBlur3D(base.IntensityAugmentationBase3D):
    r"""Apply gaussian blur given tensor image or a batch of tensor images
    randomly. The standard deviation is sampled for each instance.

    Args:
        kernel_size: the size of the kernel.
        sigma: the range for the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
            The expected modes are: ``constant``, ``reflect``, ``replicate``
            or ``circular``.
        separable: run as composition of two 1d-convolutions.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or
            broadcast it to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W, D)` or :math:`(B, C, H, W, D)`,
            Optional: :math:`(B, 4, 4)`
        - Output: :math:`(B, C, H, W, D)`
    """

    def __init__(
        self,
        kernel_size: tuple[int, int, int] | int,
        sigma: tuple[float, float] | torch.Tensor,
        border_type: str = "reflect",
        separable: bool = True,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

        self.flags = {"kernel_size": kernel_size, "separable": separable,
                      "border_type": constants.BorderType.get(border_type)}
        self._param_generator = rg.RandomGaussianBlurGenerator(sigma)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma = params["sigma"].to(
            device=input.device, dtype=input.dtype).unsqueeze(-1).expand(-1, 3)
        return gaussian_blur3d(
            input,
            self.flags["kernel_size"],
            sigma,
            self.flags["border_type"].name.lower(),
            separable=self.flags["separable"],
        )

    def compute_transformation(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: torch.Dict[str, Any]
    ) -> torch.Tensor:
        identity = torch.eye(4, dtype=input.dtype, device=input.device)[None,]
        return identity.expand(input.shape[0], 4, 4)


def _randn_like(input: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    # Generating on GPU is fastest with `torch.randn_like(...)`
    x = torch.randn_like(input)
    if std != 1.0:  # `if` is cheaper than multiplication
        x *= std
    if mean != 0.0:  # `if` is cheaper than addition
        x += mean
    return x


@gin.register(module="kornia")
class RandomGaussianNoise3D(base.IntensityAugmentationBase3D):
    r"""Add gaussian noise to a batch of multi-dimensional images.

    .. image:: _static/img/RandomGaussianNoise.png

    Args:
        mean: The mean of the gaussian distribution.
        std: The standard deviation of the gaussian distribution.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True)
            or broadcast it to the batch form (False).
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False
    ) -> None:
        super().__init__(
            p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.flags = {"mean": mean, "std": std}

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: torch.Tensor | None = None
    ) -> torch.Tensor:
        if "gaussian_noise" in params:
            gaussian_noise = params["gaussian_noise"]
        else:
            gaussian_noise = _randn_like(
                input, mean=flags["mean"], std=flags["std"])
            self._params["gaussian_noise"] = gaussian_noise
        return input + gaussian_noise

    def compute_transformation(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any]
    ) -> torch.Tensor:
        identity = torch.eye(4, dtype=input.dtype, device=input.device)[None,]
        return identity.expand(input.shape[0], 4, 4)


@gin.register(module="kornia")
class RandomGamma3D(base.IntensityAugmentationBase3D):
    r"""Apply a random transformation to the gamma of a tensor image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.

    .. image:: _static/img/RandomGamma.png

    Args:
        p: probability of applying the transformation.
        gamma: the gamma factor to apply.
        gain: the gain factor to apply.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or
            broadcast it to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W, D)` or :math:`(B, C, H, W, D)`,
            Optional: :math:`(B, 4, 4)`
        - Output: :math:`(B, C, H, W, D)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_gamma`
    """

    def __init__(
        self,
        gamma: tuple[float, float] = (1.0, 1.0),
        gain: tuple[float, float] = (1.0, 1.0),
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (gamma, "gamma_factor", None, None),
            (gain, "gain_factor", None, None)
        )

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: torch.Tensor | None = None
    ) -> torch.Tensor:
        gamma_factor = params["gamma_factor"].to(input)
        gain_factor = params["gain_factor"].to(input)
        return adjust.adjust_gamma(input, gamma_factor, gain_factor)

    def compute_transformation(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any]
    ) -> torch.Tensor:
        identity = torch.eye(4, dtype=input.dtype, device=input.device)[None,]
        return identity.expand(input.shape[0], 4, 4)
