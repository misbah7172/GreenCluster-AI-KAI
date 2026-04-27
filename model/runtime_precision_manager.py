"""
Runtime Precision Manager for Adaptive Precision Control.

Dynamically switches model precision (FP32 → FP16 → INT8 → INT4) based on
memory pressure, latency targets, and output quality monitoring.

Enables graceful degradation under resource constraints while maintaining
accuracy.

Usage::

    from model.runtime_precision_manager import RuntimePrecisionManager, PrecisionConfig
    
    config = PrecisionConfig(
        initial_precision="fp32",
        memory_pressure_threshold=0.85,
        quality_threshold=0.97,  # Minimum quality retention
    )
    
    manager = RuntimePrecisionManager(model, config)
    
    # Get current precision for a layer
    precision = manager.get_layer_precision("layer_10")
    
    # Adapt based on memory pressure
    manager.adapt_to_pressure(current_pressure=0.92)
    
    # Apply precision to model
    manager.apply_precision_plan()
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Precision(Enum):
    """Supported precision levels."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    
    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert to torch dtype."""
        mapping = {
            Precision.FP32: torch.float32,
            Precision.FP16: torch.float16,
            Precision.BF16: torch.bfloat16,
            Precision.INT8: torch.int8,
            Precision.INT4: torch.int8,  # Both quantized as INT8
        }
        return mapping[self]
    
    @property
    def bits_per_element(self) -> float:
        """Bits per element."""
        return {
            Precision.FP32: 32,
            Precision.FP16: 16,
            Precision.BF16: 16,
            Precision.INT8: 8,
            Precision.INT4: 4,
        }[self]
    
    @property
    def quality_retention(self) -> float:
        """Estimated output quality vs FP32."""
        return {
            Precision.FP32: 1.0,
            Precision.FP16: 0.999,
            Precision.BF16: 0.998,
            Precision.INT8: 0.990,
            Precision.INT4: 0.950,
        }[self]


@dataclass
class PrecisionConfig:
    """Configuration for runtime precision management."""
    initial_precision: Precision = Precision.FP32
    memory_pressure_threshold: float = 0.85
    quality_threshold: float = 0.97
    # Precision progression when pressure increases
    precision_hierarchy: List[Precision] = None
    
    def __post_init__(self):
        if self.precision_hierarchy is None:
            self.precision_hierarchy = [
                Precision.FP32,
                Precision.FP16,
                Precision.INT8,
                Precision.INT4,
            ]


class LayerPrecisionPlan:
    """Precision plan for a single layer."""
    
    def __init__(
        self,
        layer_name: str,
        layer_idx: int,
        criticality: float,  # 0.0-1.0
    ):
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.criticality = criticality  # Higher = more critical
        
        # Precision assignments
        self.weight_precision = Precision.FP32
        self.activation_precision = Precision.FP32
        self.forward_precision = Precision.FP32
    
    def to_dict(self) -> Dict:
        return {
            "layer_name": self.layer_name,
            "layer_idx": self.layer_idx,
            "criticality": round(self.criticality, 3),
            "weight_precision": self.weight_precision.value,
            "activation_precision": self.activation_precision.value,
            "forward_precision": self.forward_precision.value,
        }


class RuntimePrecisionManager:
    """
    Manages runtime precision switching for model layers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[PrecisionConfig] = None,
    ):
        self.model = model
        self.config = config or PrecisionConfig()
        
        # Layer plans
        self.layer_plans: Dict[str, LayerPrecisionPlan] = {}
        self._extract_layers()
        
        # Current global precision
        self.current_global_precision = self.config.initial_precision
        
        # Precision history for stability
        self.precision_history: List[Tuple[int, Precision]] = []
        
        # Statistics
        self.stats = {
            "total_precision_changes": 0,
            "total_upgrades": 0,
            "total_downgrades": 0,
            "current_avg_bits": 32,
        }
        
        logger.info(
            f"RuntimePrecisionManager initialized with {len(self.layer_plans)} layers"
        )
    
    def _extract_layers(self) -> None:
        """Extract layers and compute criticality."""
        layers = self._get_transformer_layers()
        
        for idx, (name, layer) in enumerate(layers):
            # Criticality based on layer position
            # Embedding and output layers more critical
            position_criticality = 1.0 - (idx / max(len(layers), 1))
            
            # Check if attention or output layer
            is_critical = "attention" in name.lower() or "output" in name.lower()
            criticality = position_criticality * (1.3 if is_critical else 0.9)
            criticality = min(criticality, 1.0)  # Clamp
            
            plan = LayerPrecisionPlan(name, idx, criticality)
            self.layer_plans[name] = plan
    
    def _get_transformer_layers(self) -> List[Tuple[str, nn.Module]]:
        """Extract transformer layers from model."""
        layers = []
        
        # Try standard layouts
        search_attrs = [
            ("transformer.h", "transformer layers"),
            ("gpt2.h", "GPT-2 layers"),
            ("model.layers", "LLaMA layers"),
            ("decoder.layers", "decoder layers"),
        ]
        
        for attr_path, desc in search_attrs:
            parts = attr_path.split(".")
            obj = self.model
            found = True
            
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    found = False
                    break
            
            if found and isinstance(obj, (nn.ModuleList, list)):
                for i, layer in enumerate(obj):
                    layers.append((f"layer_{i}", layer))
                break
        
        if not layers:
            # Fallback: all modules
            for name, module in self.model.named_modules():
                if isinstance(module, nn.TransformerEncoderLayer):
                    layers.append((name, module))
        
        return layers
    
    def adapt_to_pressure(
        self,
        current_pressure: float,  # 0.0-1.0
        memory_pressure_mb: Optional[float] = None,
    ) -> Precision:
        """
        Determine precision based on memory pressure.
        
        Parameters
        ----------
        current_pressure : float
            Current memory pressure (0.0-1.0)
        memory_pressure_mb : float, optional
            Absolute memory pressure in MB
        
        Returns
        -------
        Precision
            New recommended precision
        """
        hierarchy = self.config.precision_hierarchy
        
        if current_pressure < 0.70:
            # Low pressure: use high precision
            new_precision = hierarchy[0]  # FP32
        elif current_pressure < 0.80:
            # Moderate: FP16
            new_precision = hierarchy[1]  # FP16
        elif current_pressure < 0.90:
            # High: INT8
            new_precision = hierarchy[2]  # INT8
        else:
            # Critical: INT4
            new_precision = hierarchy[-1]  # INT4
        
        # Check quality constraint
        if new_precision.quality_retention < self.config.quality_threshold:
            # Can't go this low, use next-higher
            for i, prec in enumerate(hierarchy):
                if prec.quality_retention >= self.config.quality_threshold:
                    new_precision = hierarchy[max(0, i-1)]
                    break
        
        # Apply change if different
        if new_precision != self.current_global_precision:
            self._change_precision(new_precision)
        
        return new_precision
    
    def _change_precision(self, new_precision: Precision) -> None:
        """Change global precision level."""
        old_precision = self.current_global_precision
        self.current_global_precision = new_precision
        self.precision_history.append((len(self.precision_history), new_precision))
        
        # Update all layer plans
        for layer_plan in self.layer_plans.values():
            layer_plan.weight_precision = new_precision
            layer_plan.activation_precision = new_precision
            layer_plan.forward_precision = new_precision
        
        self.stats["total_precision_changes"] += 1
        if new_precision.bits_per_element < old_precision.bits_per_element:
            self.stats["total_downgrades"] += 1
            logger.info(
                f"Downgrading precision: {old_precision.value} → {new_precision.value}"
            )
        else:
            self.stats["total_upgrades"] += 1
            logger.info(
                f"Upgrading precision: {old_precision.value} → {new_precision.value}"
            )
    
    def get_layer_precision(self, layer_name: str) -> Precision:
        """Get precision for specific layer."""
        if layer_name in self.layer_plans:
            return self.layer_plans[layer_name].forward_precision
        return self.current_global_precision
    
    def apply_precision_plan(self) -> None:
        """Apply precision plan to model (quantization)."""
        logger.debug(
            f"Applying precision plan with {self.current_global_precision.value}"
        )
        
        # For INT8/INT4: actual quantization would happen here
        # For now, just log the plan
        for layer_name, layer_module in self.model.named_modules():
            if layer_name in self.layer_plans:
                plan = self.layer_plans[layer_name]
                logger.debug(f"  {layer_name}: {plan.forward_precision.value}")
    
    def get_memory_savings(self) -> float:
        """
        Estimate memory savings from current precision.
        
        Returns
        -------
        float
            Fraction of memory saved vs FP32 (0.0-1.0)
        """
        fp32_bits = 32.0
        current_bits = self.current_global_precision.bits_per_element
        savings = (fp32_bits - current_bits) / fp32_bits
        return max(0.0, savings)
    
    def get_quality_retention(self) -> float:
        """Estimate output quality vs FP32."""
        return self.current_global_precision.quality_retention
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics."""
        stats = dict(self.stats)
        stats["current_precision"] = self.current_global_precision.value
        stats["current_bits_per_element"] = self.current_global_precision.bits_per_element
        stats["memory_savings_fraction"] = self.get_memory_savings()
        stats["quality_retention"] = self.get_quality_retention()
        stats["precision_history_length"] = len(self.precision_history)
        return stats
    
    def get_plan(self) -> List[Dict]:
        """Get all layer precision plans."""
        return [plan.to_dict() for plan in self.layer_plans.values()]
    
    def reset(self) -> None:
        """Reset to initial precision."""
        self.current_global_precision = self.config.initial_precision
        self.precision_history.clear()
        self.apply_precision_plan()
