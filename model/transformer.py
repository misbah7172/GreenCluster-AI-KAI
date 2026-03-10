"""
Transformer model for inference benchmarking.

Implements a configurable Transformer encoder that can be used as a benchmark
workload. The model is built as a sequence of named layers so it can be
split into chunks by the chunker module.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder layer wrapped as a standalone module.

    Wraps ``nn.TransformerEncoderLayer`` so that it accepts and returns a
    plain tensor of shape ``(batch, seq_len, d_model)`` without requiring
    extra arguments. This makes it compatible with ``nn.Sequential`` and
    the chunker utility.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class InputEmbedding(nn.Module):
    """Projects integer token IDs to dense vectors and adds positional encoding."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_encoding(self.embedding(x) * self.scale)


class OutputHead(nn.Module):
    """Mean-pool the sequence and project to num_classes logits."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling over sequence
        return self.fc(x)


class TransformerModel(nn.Module):
    """Configurable Transformer encoder for inference benchmarking.

    The model is constructed as an ordered sequence of named sub-modules:
      - ``embedding``:  InputEmbedding
      - ``encoder_0`` ... ``encoder_{n-1}``:  TransformerEncoderBlock layers
      - ``output_head``:  OutputHead

    This flat structure allows the chunker to split the model at any layer
    boundary.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    d_model : int
        Embedding / hidden dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    dim_feedforward : int
        Inner dimension of the feed-forward sub-layers.
    num_classes : int
        Number of output classes.
    max_seq_len : int
        Maximum sequence length for positional encoding.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        num_classes: int = 10,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "num_classes": num_classes,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
        }

        layers = []
        layers.append(("embedding", InputEmbedding(vocab_size, d_model, max_seq_len, dropout)))
        for i in range(num_layers):
            layers.append(
                (f"encoder_{i}", TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout))
            )
        layers.append(("output_head", OutputHead(d_model, num_classes)))

        self.layers = nn.Sequential()
        for name, module in layers:
            self.layers.add_module(name, module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def get_sequential(self) -> nn.Sequential:
        """Return the internal ``nn.Sequential`` for use by the chunker."""
        return self.layers


def get_dummy_input(batch_size: int = 8, seq_len: int = 128, vocab_size: int = 30522) -> torch.Tensor:
    """Generate a random integer tensor simulating tokenized input.

    Returns
    -------
    torch.Tensor
        Shape ``(batch_size, seq_len)`` with values in ``[0, vocab_size)``.
    """
    return torch.randint(0, vocab_size, (batch_size, seq_len))


def build_model(**kwargs) -> TransformerModel:
    """Convenience factory that returns a ``TransformerModel`` in eval mode."""
    model = TransformerModel(**kwargs)
    model.eval()
    return model
