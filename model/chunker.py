"""
Model chunking utility.

Splits a PyTorch ``nn.Sequential`` model into N independent chunks
(each itself an ``nn.Sequential``). Provides helpers to serialize,
load, and inspect chunks.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelChunk(nn.Module):
    """Wrapper around a subset of layers from a parent model.

    Attributes
    ----------
    chunk_id : int
        Zero-based index of this chunk.
    layer_names : list[str]
        Names of the layers contained in this chunk.
    sequential : nn.Sequential
        The actual sub-model that performs computation.
    """

    def __init__(self, chunk_id: int, layers: List[Tuple[str, nn.Module]]):
        super().__init__()
        self.chunk_id = chunk_id
        self.layer_names = [name for name, _ in layers]
        self.sequential = nn.Sequential()
        for name, module in layers:
            self.sequential.add_module(name, module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)

    def __repr__(self) -> str:
        return (
            f"ModelChunk(id={self.chunk_id}, "
            f"layers={self.layer_names})"
        )


def chunk_model(model: nn.Module, num_chunks: int) -> List[ModelChunk]:
    """Split a model into ``num_chunks`` sequential chunks.

    The model must expose a ``get_sequential()`` method that returns an
    ``nn.Sequential`` (both ``TransformerModel`` and ``CNNModel`` do).

    Parameters
    ----------
    model : nn.Module
        A model with a ``get_sequential()`` method.
    num_chunks : int
        Number of chunks to split the model into.  Must be >= 1 and
        <= the number of layers in the model.

    Returns
    -------
    list[ModelChunk]
        Ordered list of chunks.  Feeding the output of chunk *i* as input
        to chunk *i+1* reproduces the full model's forward pass.

    Raises
    ------
    ValueError
        If ``num_chunks`` is invalid.
    """
    sequential: nn.Sequential = model.get_sequential()
    named_children = list(sequential.named_children())
    total_layers = len(named_children)

    if num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if num_chunks > total_layers:
        raise ValueError(
            f"num_chunks ({num_chunks}) exceeds total layers ({total_layers})"
        )

    # Distribute layers as evenly as possible across chunks.
    base_size = total_layers // num_chunks
    remainder = total_layers % num_chunks

    chunks: List[ModelChunk] = []
    start = 0
    for i in range(num_chunks):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        chunk_layers = named_children[start:end]
        chunks.append(ModelChunk(chunk_id=i, layers=chunk_layers))
        logger.info(
            "Chunk %d: layers %s (indices %d-%d)", i,
            [n for n, _ in chunk_layers], start, end - 1,
        )
        start = end

    return chunks


def serialize_chunk(chunk: ModelChunk, output_dir: str) -> str:
    """Save a chunk's state dict and metadata to disk.

    Creates two files inside ``output_dir``:
      - ``chunk_{id}.pt``   -- PyTorch state dict
      - ``chunk_{id}.json`` -- metadata (chunk_id, layer_names)

    Parameters
    ----------
    chunk : ModelChunk
        The chunk to save.
    output_dir : str
        Directory to write files into (created if needed).

    Returns
    -------
    str
        Path to the saved ``.pt`` file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pt_path = out / f"chunk_{chunk.chunk_id}.pt"
    meta_path = out / f"chunk_{chunk.chunk_id}.json"

    torch.save(chunk.state_dict(), pt_path)

    metadata = {
        "chunk_id": chunk.chunk_id,
        "layer_names": chunk.layer_names,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved chunk %d to %s", chunk.chunk_id, pt_path)
    return str(pt_path)


def load_chunk(
    model: nn.Module,
    chunk_id: int,
    num_chunks: int,
    weights_dir: str,
) -> ModelChunk:
    """Re-create a specific chunk from a model and load saved weights.

    Parameters
    ----------
    model : nn.Module
        The full model (used to reconstruct the chunk architecture).
    chunk_id : int
        Which chunk to load.
    num_chunks : int
        Total number of chunks the model was split into.
    weights_dir : str
        Directory containing the saved ``.pt`` file.

    Returns
    -------
    ModelChunk
        The chunk with loaded weights, in eval mode.
    """
    chunks = chunk_model(model, num_chunks)
    chunk = chunks[chunk_id]

    pt_path = Path(weights_dir) / f"chunk_{chunk_id}.pt"
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    chunk.load_state_dict(state)
    chunk.eval()

    logger.info("Loaded chunk %d from %s", chunk_id, pt_path)
    return chunk


def get_chunk_info(chunk: ModelChunk, sample_input: Optional[torch.Tensor] = None) -> Dict:
    """Return metadata about a chunk.

    Parameters
    ----------
    chunk : ModelChunk
        The chunk to inspect.
    sample_input : torch.Tensor, optional
        If provided, a forward pass is run to capture output shape.

    Returns
    -------
    dict
        Contains ``chunk_id``, ``layer_names``, ``num_parameters``,
        and optionally ``output_shape``.
    """
    num_params = sum(p.numel() for p in chunk.parameters())
    info: Dict = {
        "chunk_id": chunk.chunk_id,
        "layer_names": chunk.layer_names,
        "num_parameters": num_params,
    }

    if sample_input is not None:
        with torch.no_grad():
            output = chunk(sample_input)
        info["output_shape"] = list(output.shape)

    return info


def validate_chunks(model: nn.Module, chunks: List[ModelChunk], sample_input: torch.Tensor, atol: float = 1e-5) -> bool:
    """Verify that chaining chunks reproduces the full model output.

    Parameters
    ----------
    model : nn.Module
        The original full model.
    chunks : list[ModelChunk]
        Ordered list of chunks produced by ``chunk_model()``.
    sample_input : torch.Tensor
        Input tensor appropriate for the model.
    atol : float
        Absolute tolerance for output comparison.

    Returns
    -------
    bool
        ``True`` if the chunk chain output matches the full model output.
    """
    model.eval()
    for c in chunks:
        c.eval()

    with torch.no_grad():
        expected = model(sample_input)

        x = sample_input
        for c in chunks:
            x = c(x)

    match = torch.allclose(expected, x, atol=atol)
    if match:
        logger.info("Chunk validation PASSED (atol=%s)", atol)
    else:
        max_diff = (expected - x).abs().max().item()
        logger.warning("Chunk validation FAILED (max diff=%s, atol=%s)", max_diff, atol)

    return match
