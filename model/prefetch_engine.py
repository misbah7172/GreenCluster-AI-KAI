"""
Double-buffered prefetching engine.

While the GPU processes Layer N using **Buffer A**, a background thread
loads Layer N+1 weights from RAM / disk into **Buffer B**.  On
completion the buffers swap roles, hiding the memory transfer latency
behind GPU computation time.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class PrefetchEngine:
    """Asynchronous double-buffered weight prefetcher.

    Parameters
    ----------
    weight_manager : TieredWeightManager
        Source for loading layer weights.
    device : str
        GPU device for the active buffer.
    """

    def __init__(self, weight_manager, device: str = "cuda:0"):
        self._weight_manager = weight_manager
        self._device = device

        self._buffer_a: Optional[Dict[str, torch.Tensor]] = None
        self._buffer_b: Optional[Dict[str, torch.Tensor]] = None
        self._active_buffer: str = "a"  # "a" or "b"

        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_ready = threading.Event()
        self._prefetch_error: Optional[Exception] = None

        # Timing statistics
        self._stats: Dict[str, float] = {
            "total_prefetch_ms": 0.0,
            "total_swap_ms": 0.0,
            "total_gpu_idle_ms": 0.0,
            "prefetch_count": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefetch_layer(self, layer_name: str) -> None:
        """Start async loading of *layer_name* into the inactive buffer.

        Non-blocking.  The inactive buffer is filled in a background
        thread.  Call :meth:`wait_and_swap` to block until the prefetch
        completes and activate the new buffer.
        """
        self._prefetch_ready.clear()
        self._prefetch_error = None

        self._prefetch_thread = threading.Thread(
            target=self._do_prefetch,
            args=(layer_name,),
            daemon=True,
            name=f"prefetch-{layer_name}",
        )
        self._prefetch_thread.start()

    def wait_and_swap(self) -> Optional[Dict[str, torch.Tensor]]:
        """Block until prefetch completes, then swap buffers.

        Returns the now-active buffer's state dict (on GPU), or
        ``None`` if prefetch failed.
        """
        idle_start = time.perf_counter()
        self._prefetch_ready.wait()
        idle_ms = (time.perf_counter() - idle_start) * 1000.0
        self._stats["total_gpu_idle_ms"] += idle_ms

        if self._prefetch_error is not None:
            logger.error("Prefetch failed: %s", self._prefetch_error)
            return None

        swap_start = time.perf_counter()
        # Swap active buffer pointer
        if self._active_buffer == "a":
            self._active_buffer = "b"
            result = self._buffer_b
        else:
            self._active_buffer = "a"
            result = self._buffer_a
        swap_ms = (time.perf_counter() - swap_start) * 1000.0
        self._stats["total_swap_ms"] += swap_ms

        return result

    def get_active_buffer(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return the current active buffer without swapping."""
        if self._active_buffer == "a":
            return self._buffer_a
        return self._buffer_b

    def get_prefetch_stats(self) -> Dict[str, Any]:
        """Return timing stats.

        Returns
        -------
        dict
            ``total_prefetch_ms``, ``total_swap_ms``, ``total_gpu_idle_ms``,
            ``prefetch_count``, ``avg_prefetch_ms``, ``overlap_pct``.
        """
        stats = dict(self._stats)
        count = stats["prefetch_count"]

        stats["avg_prefetch_ms"] = (
            stats["total_prefetch_ms"] / count if count > 0 else 0.0
        )

        total_transfer = stats["total_prefetch_ms"]
        if total_transfer > 0:
            overlap = max(0.0, total_transfer - stats["total_gpu_idle_ms"])
            stats["overlap_pct"] = round(overlap / total_transfer * 100.0, 2)
        else:
            stats["overlap_pct"] = 0.0

        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_prefetch(self, layer_name: str) -> None:
        """Background: load layer weights into the inactive buffer."""
        prefetch_start = time.perf_counter()
        try:
            from model.tiered_weight_manager import StorageTier

            # Promote layer to GPU VRAM via the weight manager
            state_dict = self._weight_manager.promote_layer(
                layer_name, StorageTier.GPU_VRAM,
            )

            if state_dict is None:
                # Fallback: try loading from disk directly
                state_dict = self._weight_manager.load_layer_from_disk(
                    layer_name, device=self._device,
                )

            # Place into inactive buffer
            if self._active_buffer == "a":
                self._buffer_b = state_dict
            else:
                self._buffer_a = state_dict

            prefetch_ms = (time.perf_counter() - prefetch_start) * 1000.0
            self._stats["total_prefetch_ms"] += prefetch_ms
            self._stats["prefetch_count"] += 1

            logger.debug(
                "Prefetched %s in %.2f ms", layer_name, prefetch_ms,
            )

        except Exception as e:
            self._prefetch_error = e
            logger.exception("Prefetch error for %s", layer_name)

        finally:
            self._prefetch_ready.set()
