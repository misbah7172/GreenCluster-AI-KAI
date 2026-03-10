"""
gRPC server for serving a single model chunk.

Each instance loads one chunk of a split model and exposes it via the
InferenceService defined in proto/inference.proto.  The chunk ID and
port are configured through environment variables or constructor args.

Usage (standalone)::

    # Requires chunk weights at <weights_dir>/chunk_<id>.pt
    python -m model.chunk_server \
        --chunk-id 0 \
        --num-chunks 3 \
        --model-type transformer \
        --weights-dir ./weights \
        --port 50051

Environment variable overrides (for container deployments)::

    CHUNK_ID, NUM_CHUNKS, MODEL_TYPE, WEIGHTS_DIR, PORT
"""

import argparse
import io
import logging
import os
import sys
import time
from concurrent import futures

import torch
import grpc

# Ensure project root is importable when running as __main__
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from proto import inference_pb2
from proto import inference_pb2_grpc
from model.chunker import chunk_model, load_chunk

logger = logging.getLogger(__name__)


def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes using torch.save into an in-memory buffer."""
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def _deserialize_tensor(data: bytes) -> torch.Tensor:
    """Deserialize bytes back to a tensor."""
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)


def _build_model(model_type: str):
    """Build and return a model in eval mode."""
    if model_type == "transformer":
        from model.transformer import build_model
        return build_model()
    elif model_type == "cnn":
        from model.cnn import build_model
        return build_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC servicer that runs inference on a single model chunk."""

    def __init__(
        self,
        chunk_id: int,
        num_chunks: int,
        model_type: str,
        weights_dir: str,
        device: str = "cpu",
    ):
        super().__init__()
        self.chunk_id = chunk_id
        self.device = torch.device(device)

        logger.info(
            "Loading chunk %d/%d (model=%s, weights=%s, device=%s)",
            chunk_id, num_chunks, model_type, weights_dir, device,
        )

        model = _build_model(model_type)

        if weights_dir and os.path.isdir(weights_dir):
            self.chunk = load_chunk(model, chunk_id, num_chunks, weights_dir)
        else:
            # No pre-saved weights — split the model on the fly (useful for testing)
            chunks = chunk_model(model, num_chunks)
            self.chunk = chunks[chunk_id]

        self.chunk = self.chunk.to(self.device)
        self.chunk.eval()
        self._ready = True
        logger.info("Chunk %d ready: %s", chunk_id, self.chunk.layer_names)

    def Infer(self, request, context):
        """Run forward pass on the chunk and return the output tensor."""
        start = time.perf_counter()

        input_tensor = _deserialize_tensor(request.tensor_data)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output_tensor = self.chunk(input_tensor)

        output_tensor = output_tensor.cpu()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        response = inference_pb2.InferResponse(
            tensor_data=_serialize_tensor(output_tensor),
            chunk_id=self.chunk_id,
            request_id=request.request_id,
            tensor_shape=list(output_tensor.shape),
            tensor_dtype=str(output_tensor.dtype).replace("torch.", ""),
            inference_time_ms=elapsed_ms,
        )

        logger.debug(
            "Chunk %d processed request %s in %.2f ms (output shape %s)",
            self.chunk_id, request.request_id, elapsed_ms, list(output_tensor.shape),
        )
        return response

    def HealthCheck(self, request, context):
        """Return readiness status."""
        return inference_pb2.HealthResponse(
            ready=self._ready,
            chunk_id=self.chunk_id,
            status="ready" if self._ready else "not ready",
        )


def serve(
    chunk_id: int,
    num_chunks: int,
    model_type: str,
    weights_dir: str,
    port: int,
    device: str = "cpu",
    max_workers: int = 4,
    max_message_length: int = 256 * 1024 * 1024,  # 256 MB
):
    """Start the gRPC server for a model chunk.

    Parameters
    ----------
    chunk_id : int
        Zero-based chunk index.
    num_chunks : int
        Total number of chunks.
    model_type : str
        ``"transformer"`` or ``"cnn"``.
    weights_dir : str
        Directory containing chunk weight files.
    port : int
        Port to listen on.
    device : str
        Torch device (e.g. ``"cpu"``, ``"cuda:0"``).
    max_workers : int
        Thread pool size for the gRPC server.
    max_message_length : int
        Maximum gRPC message size in bytes.
    """
    options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=options,
    )

    servicer = InferenceServicer(
        chunk_id=chunk_id,
        num_chunks=num_chunks,
        model_type=model_type,
        weights_dir=weights_dir,
        device=device,
    )

    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("Chunk %d serving on port %d", chunk_id, port)

    return server


def _env_or(name: str, default):
    """Read from environment variable or return default."""
    val = os.environ.get(name)
    if val is None:
        return default
    return type(default)(val)


def main():
    parser = argparse.ArgumentParser(description="Model chunk gRPC server")
    parser.add_argument("--chunk-id", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    chunk_id = args.chunk_id if args.chunk_id is not None else _env_or("CHUNK_ID", 0)
    num_chunks = args.num_chunks if args.num_chunks is not None else _env_or("NUM_CHUNKS", 2)
    model_type = args.model_type if args.model_type is not None else os.environ.get("MODEL_TYPE", "transformer")
    weights_dir = args.weights_dir if args.weights_dir is not None else os.environ.get("WEIGHTS_DIR", "")
    port = args.port if args.port is not None else _env_or("PORT", 50051)
    device = args.device if args.device is not None else os.environ.get("DEVICE", "cpu")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [chunk-%(name)s] %(levelname)s %(message)s",
    )

    server = serve(
        chunk_id=chunk_id,
        num_chunks=num_chunks,
        model_type=model_type,
        weights_dir=weights_dir,
        port=port,
        device=device,
    )

    logger.info("Press Ctrl+C to stop.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down chunk %d server.", chunk_id)
        server.stop(grace=5)


if __name__ == "__main__":
    main()
