from torch import nn


def get_conv_nd(dim: int, transposed: bool = False) -> nn.modules.conv._ConvNd:
    if not transposed:
        return getattr(nn, f"Conv{dim}d")
    elif transposed:
        return getattr(nn, f"ConvTranspose{dim}d")


def get_maxpool_nd(dim: int) -> nn.modules.pooling._MaxPoolNd:
    return getattr(nn, f"MaxPool{dim}d")


def get_adaptive_averagepool_nd(dim: int) -> nn.modules.pooling._MaxPoolNd:
    return getattr(nn, f"AdaptiveAvgPool{dim}d")
