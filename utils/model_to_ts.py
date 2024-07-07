""" Convert model to torchscript."""

import argparse
from collections import OrderedDict

import gin
from torch import nn
import torch.jit


def load_model_state_dict(ckpt_path: str):
    """Load model state dict from lightning checkpoint."""
    state_dict = torch.load(
        ckpt_path, map_location=torch.device("cpu"))['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "model." not in k:
            continue
        name = k.replace("model.", "", 1)
        new_state_dict[name] = v
    return new_state_dict


@gin.configurable
def model_to_ts(
    model: nn.Module,
    ckpt_path: str,
    ts_file: str,
    in_channels: int = 1,
    input_shape: tuple[int, int, int] = (64, 64, 64),
) -> None:
    """Convert model to torchscript.

    Args:
        model (nn.Module): Model to convert to torchscript.
        ckpt_path (str): Path to lightning checkpoint.
        ts_file (str): Path to save torchscript model.
        input_shape (tuple[int, int, int]): Input shape.
    """
    model = model.eval()
    state_dict = load_model_state_dict(ckpt_path)
    model.load_state_dict(state_dict)
    inputs = torch.randn(2, in_channels, *input_shape)
    model = torch.jit.trace(model, inputs)
    torch.jit.save(model, ts_file)


def main():
    """ Main function. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="config.gin file", type=str, required=True)
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    model_to_ts()


if __name__ == "__main__":
    main()
