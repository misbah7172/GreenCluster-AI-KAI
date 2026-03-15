"""
Tiered Weight Manager for FlexGen-style CPU offloading.

Manages model weights across three storage tiers:

* **Tier 0 — GPU_VRAM**: Active transformer blocks undergoing a forward pass.
* **Tier 1 — SYSTEM_RAM**: Inactive layers cached for rapid promotion.
* **Tier 2 — DISK_SWAP**: Weights stored as safetensors on disk for models
  exceeding total system RAM.

Safetensors format is mandatory for disk I/O to minimise serialisation
energy.
"""

import logging
import os
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Optional safetensors import — required for disk tier
try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load
    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False
    logger.warning("safetensors not available — disk tier disabled")


class StorageTier(Enum):
    """Storage tier for model weights."""
    GPU_VRAM = 0
    SYSTEM_RAM = 1
    DISK_SWAP = 2


@dataclass
class LayerPlacement:
    """Tracks where a specific layer's weights currently reside."""
    layer_name: str
    current_tier: StorageTier
    size_mb: float
    disk_path: Optional[str] = None
    is_loaded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "current_tier": self.current_tier.name,
            "size_mb": round(self.size_mb, 2),
            "disk_path": self.disk_path,
            "is_loaded": self.is_loaded,
        }


@dataclass
class TierCapacity:
    """Capacity tracking for each storage tier."""
    tier: StorageTier
    total_mb: float
    used_mb: float = 0.0

    @property
    def free_mb(self) -> float:
        return max(0.0, self.total_mb - self.used_mb)


class TieredWeightManager:
    """Manages layer weights across GPU VRAM, System RAM, and Disk.

    Parameters
    ----------
    gpu_budget_mb : float
        VRAM available for model weights (after CUDA overhead).
    ram_budget_mb : float
        RAM available for weight caching.
    disk_dir : str
        Directory for disk-swapped safetensors files.
    device : str
        GPU device string (e.g. ``"cuda:0"``).
    """

    def __init__(
        self,
        gpu_budget_mb: float,
        ram_budget_mb: float,
        disk_dir: str = "/tmp/kai_swap",
        device: str = "cuda:0",
    ):
        self._device = device
        self._disk_dir = disk_dir
        Path(disk_dir).mkdir(parents=True, exist_ok=True)

        self._tiers = {
            StorageTier.GPU_VRAM: TierCapacity(StorageTier.GPU_VRAM, gpu_budget_mb),
            StorageTier.SYSTEM_RAM: TierCapacity(StorageTier.SYSTEM_RAM, ram_budget_mb),
            StorageTier.DISK_SWAP: TierCapacity(StorageTier.DISK_SWAP, float("inf")),
        }

        # layer_name -> LayerPlacement
        self._placements: Dict[str, LayerPlacement] = {}
        # layer_name -> state_dict on CPU (for RAM tier)
        self._ram_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Placement planning
    # ------------------------------------------------------------------

    def plan_placement(
        self,
        layer_sizes_mb: List[Tuple[str, float]],
    ) -> Dict[str, LayerPlacement]:
        """Decide which tier each layer goes to.

        Algorithm: fill GPU VRAM first, then RAM, remainder to disk.

        Parameters
        ----------
        layer_sizes_mb : list[(name, size_mb)]
            Layers in forward-pass order with estimated sizes.

        Returns
        -------
        dict[str, LayerPlacement]
            Placement map keyed by layer name.
        """
        gpu_cap = self._tiers[StorageTier.GPU_VRAM]
        ram_cap = self._tiers[StorageTier.SYSTEM_RAM]

        placements: Dict[str, LayerPlacement] = {}

        for name, size in layer_sizes_mb:
            if gpu_cap.free_mb >= size:
                tier = StorageTier.GPU_VRAM
                gpu_cap.used_mb += size
            elif ram_cap.free_mb >= size:
                tier = StorageTier.SYSTEM_RAM
                ram_cap.used_mb += size
            else:
                tier = StorageTier.DISK_SWAP

            disk_path = None
            if tier == StorageTier.DISK_SWAP:
                disk_path = os.path.join(self._disk_dir, f"{name}.safetensors")

            placements[name] = LayerPlacement(
                layer_name=name,
                current_tier=tier,
                size_mb=size,
                disk_path=disk_path,
                is_loaded=(tier != StorageTier.DISK_SWAP),
            )

        self._placements = placements
        return dict(placements)

    # ------------------------------------------------------------------
    # Disk I/O (safetensors only)
    # ------------------------------------------------------------------

    def store_layer_to_disk(
        self,
        layer_name: str,
        state_dict: Dict[str, torch.Tensor],
    ) -> str:
        """Save a layer's state dict to disk as safetensors.

        Returns the path to the saved file.
        """
        if not _SAFETENSORS_AVAILABLE:
            raise RuntimeError("safetensors is required for disk tier")

        path = os.path.join(self._disk_dir, f"{layer_name}.safetensors")
        # Ensure all tensors are contiguous and on CPU
        cpu_dict = {k: v.contiguous().cpu() for k, v in state_dict.items()}
        _st_save(cpu_dict, path)

        if layer_name in self._placements:
            self._placements[layer_name].disk_path = path
            self._placements[layer_name].current_tier = StorageTier.DISK_SWAP
            self._placements[layer_name].is_loaded = False

        logger.debug("Stored %s to disk: %s", layer_name, path)
        return path

    def load_layer_from_disk(
        self,
        layer_name: str,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Load a layer from disk safetensors file.

        Parameters
        ----------
        layer_name : str
            Layer identifier.
        device : str
            Target device (``"cpu"`` or ``"cuda:0"``).

        Returns
        -------
        dict[str, Tensor]
            State dict with tensors on *device*.
        """
        if not _SAFETENSORS_AVAILABLE:
            raise RuntimeError("safetensors is required for disk tier")

        placement = self._placements.get(layer_name)
        if placement is None or placement.disk_path is None:
            raise FileNotFoundError(f"No disk path for layer {layer_name}")

        state_dict = _st_load(placement.disk_path, device=device)
        return state_dict

    # ------------------------------------------------------------------
    # Tier transitions
    # ------------------------------------------------------------------

    def promote_layer(
        self,
        layer_name: str,
        target_tier: StorageTier,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Move a layer from a lower tier to a higher tier.

        Returns the state dict on the target device, or ``None`` if
        promotion is not possible.
        """
        placement = self._placements.get(layer_name)
        if placement is None:
            return None

        if target_tier.value >= placement.current_tier.value:
            # Not a promotion (same or lower tier)
            return self._ram_cache.get(layer_name)

        state_dict = None

        if placement.current_tier == StorageTier.DISK_SWAP:
            # Promote disk → RAM or GPU
            device = "cpu" if target_tier == StorageTier.SYSTEM_RAM else self._device
            state_dict = self.load_layer_from_disk(layer_name, device=device)
            if target_tier == StorageTier.SYSTEM_RAM:
                self._ram_cache[layer_name] = state_dict

        elif placement.current_tier == StorageTier.SYSTEM_RAM:
            # Promote RAM → GPU
            cached = self._ram_cache.get(layer_name)
            if cached is not None:
                state_dict = {
                    k: v.to(self._device) for k, v in cached.items()
                }

        if state_dict is not None:
            target_cap = self._tiers[target_tier]
            target_cap.used_mb += placement.size_mb

            src_cap = self._tiers[placement.current_tier]
            src_cap.used_mb = max(0, src_cap.used_mb - placement.size_mb)

            placement.current_tier = target_tier
            placement.is_loaded = True

        return state_dict

    def demote_layer(
        self,
        layer_name: str,
        target_tier: StorageTier,
    ) -> None:
        """Move a layer from a higher tier to a lower tier."""
        placement = self._placements.get(layer_name)
        if placement is None:
            return

        if target_tier.value <= placement.current_tier.value:
            return  # not a demotion

        if target_tier == StorageTier.DISK_SWAP:
            # Must store to disk first if not already there
            cached = self._ram_cache.pop(layer_name, None)
            if cached is not None and placement.disk_path is None:
                self.store_layer_to_disk(layer_name, cached)

        elif target_tier == StorageTier.SYSTEM_RAM:
            # GPU → RAM: move tensors to CPU and cache
            pass  # caller should provide the state dict

        src_cap = self._tiers[placement.current_tier]
        src_cap.used_mb = max(0, src_cap.used_mb - placement.size_mb)

        placement.current_tier = target_tier
        placement.is_loaded = (target_tier != StorageTier.DISK_SWAP)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_placement_summary(self) -> Dict[str, Any]:
        """Return tier occupancy and per-layer placement."""
        return {
            "tiers": self.get_tier_stats(),
            "layers": {
                name: p.to_dict() for name, p in self._placements.items()
            },
        }

    def get_tier_stats(self) -> List[Dict[str, Any]]:
        """Return capacity / usage for each tier."""
        return [
            {
                "tier": cap.tier.name,
                "total_mb": round(cap.total_mb, 2) if cap.total_mb != float("inf") else "unlimited",
                "used_mb": round(cap.used_mb, 2),
                "free_mb": round(cap.free_mb, 2) if cap.free_mb != float("inf") else "unlimited",
            }
            for cap in self._tiers.values()
        ]
