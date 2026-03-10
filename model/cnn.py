"""
CNN model for inference benchmarking.

Implements a configurable convolutional neural network that can be used as
an alternative benchmark workload. The model is built as a sequence of
named layers so it can be split into chunks by the chunker module.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PoolBlock(nn.Module):
    """Adaptive average pooling followed by flattening."""

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x).flatten(1)


class Classifier(nn.Module):
    """Fully connected classifier head."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CNNModel(nn.Module):
    """Configurable CNN for inference benchmarking.

    Architecture (default 6-block config):
      - ``conv_0``:  ConvBlock  3 -> 64
      - ``conv_1``:  ConvBlock  64 -> 64,  stride 2
      - ``conv_2``:  ConvBlock  64 -> 128
      - ``conv_3``:  ConvBlock  128 -> 128, stride 2
      - ``conv_4``:  ConvBlock  128 -> 256
      - ``conv_5``:  ConvBlock  256 -> 256, stride 2
      - ``pool``:    PoolBlock  (adaptive avg pool + flatten)
      - ``classifier``: Classifier  256 -> num_classes

    The model is exposed as ``nn.Sequential`` via ``get_sequential()``
    for compatibility with the chunker.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (e.g. 3 for RGB).
    num_classes : int
        Number of output classes.
    base_channels : int
        Number of channels in the first conv layer; doubled every 2 layers.
    num_conv_layers : int
        Total number of conv blocks (must be even).
    dropout : float
        Dropout probability in the classifier head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        num_conv_layers: int = 6,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "base_channels": base_channels,
            "num_conv_layers": num_conv_layers,
            "dropout": dropout,
        }

        layers = []
        ch_in = in_channels
        ch_out = base_channels

        for i in range(num_conv_layers):
            stride = 2 if (i % 2 == 1) else 1
            layers.append((f"conv_{i}", ConvBlock(ch_in, ch_out, kernel_size=3, stride=stride)))
            ch_in = ch_out
            if i % 2 == 1 and ch_out < 512:
                ch_out = ch_out * 2

        layers.append(("pool", PoolBlock(output_size=1)))
        layers.append(("classifier", Classifier(ch_in, num_classes, dropout)))

        self.layers = nn.Sequential()
        for name, module in layers:
            self.layers.add_module(name, module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def get_sequential(self) -> nn.Sequential:
        """Return the internal ``nn.Sequential`` for use by the chunker."""
        return self.layers


def get_dummy_input(batch_size: int = 8, channels: int = 3, height: int = 224, width: int = 224) -> torch.Tensor:
    """Generate a random float tensor simulating an image batch.

    Returns
    -------
    torch.Tensor
        Shape ``(batch_size, channels, height, width)`` with values in ``[0, 1)``.
    """
    return torch.rand(batch_size, channels, height, width)


def build_model(**kwargs) -> CNNModel:
    """Convenience factory that returns a ``CNNModel`` in eval mode."""
    model = CNNModel(**kwargs)
    model.eval()
    return model
