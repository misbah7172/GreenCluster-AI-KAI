"""
Adaptive Precision Controller for KAI.

Implements dynamic precision adjustment based on layer criticality,
memory pressure, and power usage. Maintains output quality while
optimizing resource utilization.

Usage::

    from model.adaptive_precision import AdaptivePrecisionController
    
    controller = AdaptivePrecisionController(
        memory_threshold=0.85,
        power_threshold=0.80,
    )
    
    # Analyze model and get precision recommendations
    plan = controller.analyze_model(model)
    
    # Apply adaptive precision
    optimized_model = controller.apply_precision(model, plan)
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from model.plugin_architecture import PluginRegistry, OptimizerPlugin

logger = logging.getLogger(__name__)


class PrecisionLevel(Enum):
    """Supported precision levels."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    
    @property
    def bytes_per_element(self) -> float:
        """Memory cost per element."""
        return {
            PrecisionLevel.FP32: 4.0,
            PrecisionLevel.FP16: 2.0,
            PrecisionLevel.BF16: 2.0,
            PrecisionLevel.INT8: 1.0,
            PrecisionLevel.INT4: 0.5,
        }[self]
    
    @property
    def relative_quality(self) -> float:
        """Relative output quality (1.0 = perfect)."""
        return {
            PrecisionLevel.FP32: 1.0,
            PrecisionLevel.FP16: 0.999,
            PrecisionLevel.BF16: 0.998,
            PrecisionLevel.INT8: 0.995,
            PrecisionLevel.INT4: 0.985,
        }[self]


class LayerType(Enum):
    """Classification of layer types for precision decisions."""
    ATTENTION = "attention"
    FEED_FORWARD = "feed_forward"
    EMBEDDING = "embedding"
    NORM = "norm"
    OUTPUT = "output"
    UNKNOWN = "unknown"


@dataclass
class LayerCriticality:
    """Criticality assessment for a single layer."""
    layer_name: str
    layer_type: LayerType
    criticality_score: float  # 0.0-1.0, higher = more critical
    recommended_precision: PrecisionLevel
    memory_mb: float
    parameter_count: int
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "layer_type": self.layer_type.value,
            "criticality_score": round(self.criticality_score, 3),
            "recommended_precision": self.recommended_precision.value,
            "memory_mb": round(self.memory_mb, 2),
            "parameter_count": self.parameter_count,
            "reasons": self.reasons,
        }


@dataclass
class PrecisionPlan:
    """Complete precision plan for a model."""
    model_name: str
    total_params: int
    original_memory_mb: float
    optimized_memory_mb: float
    layer_plans: List[LayerCriticality]
    memory_savings_pct: float
    estimated_quality_retention: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_params": self.total_params,
            "original_memory_mb": round(self.original_memory_mb, 2),
            "optimized_memory_mb": round(self.optimized_memory_mb, 2),
            "memory_savings_pct": round(self.memory_savings_pct, 2),
            "estimated_quality_retention": round(self.estimated_quality_retention, 4),
            "layers": [lp.to_dict() for lp in self.layer_plans],
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Precision Plan for {self.model_name}",
            f"  Total parameters: {self.total_params:,}",
            f"  Original memory: {self.original_memory_mb:.0f} MB",
            f"  Optimized memory: {self.optimized_memory_mb:.0f} MB",
            f"  Memory savings: {self.memory_savings_pct:.1f}%",
            f"  Est. quality retention: {self.estimated_quality_retention*100:.2f}%",
            "",
            "Layer Precision Assignments:",
        ]
        
        for lp in self.layer_plans:
            lines.append(
                f"  {lp.layer_name}: {lp.recommended_precision.value} "
                f"(criticality={lp.criticality_score:.2f})"
            )
        
        return "\n".join(lines)


class AdaptivePrecisionController:
    """Dynamically adjusts precision based on runtime conditions.
    
    Parameters
    ----------
    memory_threshold : float
        Memory pressure threshold (0.0-1.0) to trigger lower precision.
    power_threshold : float
        Power usage threshold (fraction of TDP) to trigger lower precision.
    min_quality : float
        Minimum acceptable quality retention (0.0-1.0).
    critical_layer_types : set
        Layer types that should maintain higher precision.
    """
    
    # Layers that are critical for output quality
    CRITICAL_PATTERNS = {
        "output", "lm_head", "embed", "final", "cls", "classifier",
        "layer_norm", "ln_f", "ln_1", "ln_2",
    }
    
    # Layers safe for aggressive quantization
    SAFE_PATTERNS = {
        "fc", "mlp", "dense", "linear", "ffn", "intermediate",
    }
    
    def __init__(
        self,
        memory_threshold: float = 0.85,
        power_threshold: float = 0.80,
        min_quality: float = 0.98,
        critical_layer_types: Optional[Set[LayerType]] = None,
    ):
        self._memory_threshold = memory_threshold
        self._power_threshold = power_threshold
        self._min_quality = min_quality
        self._critical_types = critical_layer_types or {
            LayerType.EMBEDDING,
            LayerType.OUTPUT,
            LayerType.NORM,
        }
        
        # Runtime state
        self._current_memory_pressure = 0.0
        self._current_power_pct = 0.0
        self._layer_precisions: Dict[str, PrecisionLevel] = {}
    
    def analyze_model(
        self,
        model: nn.Module,
        base_precision: PrecisionLevel = PrecisionLevel.FP16,
    ) -> PrecisionPlan:
        """Analyze model and generate precision recommendations.
        
        Parameters
        ----------
        model : nn.Module
            Model to analyze
        base_precision : PrecisionLevel
            Starting precision for the model
            
        Returns
        -------
        PrecisionPlan
            Recommended precision for each layer
        """
        layer_plans: List[LayerCriticality] = []
        total_params = 0
        original_memory = 0.0
        optimized_memory = 0.0
        
        model_name = model.__class__.__name__
        
        for name, module in model.named_modules():
            if not list(module.children()):  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters())
                if param_count == 0:
                    continue
                
                total_params += param_count
                layer_type = self._classify_layer(name, module)
                criticality = self._compute_criticality(name, module, layer_type)
                
                # Determine recommended precision
                recommended = self._recommend_precision(
                    layer_type, criticality, base_precision
                )
                
                # Memory calculations
                original_mb = (param_count * base_precision.bytes_per_element) / (1024**2)
                optimized_mb = (param_count * recommended.bytes_per_element) / (1024**2)
                
                original_memory += original_mb
                optimized_memory += optimized_mb
                
                reasons = self._get_precision_reasons(
                    layer_type, criticality, recommended
                )
                
                layer_plans.append(LayerCriticality(
                    layer_name=name,
                    layer_type=layer_type,
                    criticality_score=criticality,
                    recommended_precision=recommended,
                    memory_mb=optimized_mb,
                    parameter_count=param_count,
                    reasons=reasons,
                ))
        
        # Calculate quality retention
        quality = self._estimate_quality_retention(layer_plans)
        
        savings_pct = ((original_memory - optimized_memory) / original_memory * 100
                       if original_memory > 0 else 0.0)
        
        return PrecisionPlan(
            model_name=model_name,
            total_params=total_params,
            original_memory_mb=original_memory,
            optimized_memory_mb=optimized_memory,
            layer_plans=layer_plans,
            memory_savings_pct=savings_pct,
            estimated_quality_retention=quality,
        )
    
    def apply_precision(
        self,
        model: nn.Module,
        plan: PrecisionPlan,
        device: str = "cuda:0",
    ) -> nn.Module:
        """Apply precision plan to model.
        
        Parameters
        ----------
        model : nn.Module
            Model to optimize
        plan : PrecisionPlan
            Precision assignments
        device : str
            Target device
            
        Returns
        -------
        nn.Module
            Model with mixed precision applied
        """
        # Create mapping of layer names to precision
        precision_map = {
            lp.layer_name: lp.recommended_precision
            for lp in plan.layer_plans
        }
        
        # Apply precision to each layer
        for name, module in model.named_modules():
            if name in precision_map:
                precision = precision_map[name]
                self._apply_layer_precision(module, precision, device)
                self._layer_precisions[name] = precision
        
        logger.info(
            "Applied precision plan: %d layers, %.1f%% memory savings",
            len(plan.layer_plans), plan.memory_savings_pct
        )
        
        return model
    
    def update_runtime_conditions(
        self,
        memory_pressure: Optional[float] = None,
        power_pct: Optional[float] = None,
    ) -> None:
        """Update runtime conditions for adaptive decisions.
        
        Parameters
        ----------
        memory_pressure : float, optional
            Current memory usage as fraction (0.0-1.0)
        power_pct : float, optional
            Current power as fraction of TDP (0.0-1.0)
        """
        if memory_pressure is not None:
            self._current_memory_pressure = memory_pressure
        if power_pct is not None:
            self._current_power_pct = power_pct
    
    def should_reduce_precision(self) -> bool:
        """Check if conditions warrant reducing precision."""
        return (
            self._current_memory_pressure > self._memory_threshold or
            self._current_power_pct > self._power_threshold
        )
    
    def get_dynamic_precision(
        self,
        layer_name: str,
        base_precision: PrecisionLevel,
    ) -> PrecisionLevel:
        """Get precision for a layer considering runtime conditions.
        
        Parameters
        ----------
        layer_name : str
            Name of the layer
        base_precision : PrecisionLevel
            Default precision for the layer
            
        Returns
        -------
        PrecisionLevel
            Adjusted precision based on runtime conditions
        """
        if not self.should_reduce_precision():
            return base_precision
        
        # Check if layer is critical
        is_critical = any(
            pattern in layer_name.lower()
            for pattern in self.CRITICAL_PATTERNS
        )
        
        if is_critical:
            # Critical layers: at most go to FP16
            if base_precision == PrecisionLevel.FP32:
                return PrecisionLevel.FP16
            return base_precision
        
        # Non-critical: can be more aggressive
        precision_order = [
            PrecisionLevel.FP32,
            PrecisionLevel.FP16,
            PrecisionLevel.BF16,
            PrecisionLevel.INT8,
            PrecisionLevel.INT4,
        ]
        
        current_idx = precision_order.index(base_precision)
        
        # Determine step based on pressure
        pressure = max(self._current_memory_pressure, self._current_power_pct)
        if pressure > 0.95:
            step = 2
        elif pressure > 0.90:
            step = 1
        else:
            step = 0
        
        new_idx = min(current_idx + step, len(precision_order) - 1)
        return precision_order[new_idx]
    
    # -----------------------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------------------
    
    def _classify_layer(self, name: str, module: nn.Module) -> LayerType:
        """Classify a layer based on name and type."""
        name_lower = name.lower()
        
        # Check name patterns
        if any(p in name_lower for p in ["attn", "attention", "self_attn"]):
            return LayerType.ATTENTION
        if any(p in name_lower for p in ["mlp", "ffn", "feed_forward", "fc", "dense"]):
            return LayerType.FEED_FORWARD
        if any(p in name_lower for p in ["embed", "wte", "wpe"]):
            return LayerType.EMBEDDING
        if any(p in name_lower for p in ["norm", "ln_"]):
            return LayerType.NORM
        if any(p in name_lower for p in ["lm_head", "output", "cls", "classifier"]):
            return LayerType.OUTPUT
        
        # Check module type
        if isinstance(module, nn.Embedding):
            return LayerType.EMBEDDING
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            return LayerType.NORM
        if isinstance(module, nn.Linear):
            if "out" in name_lower or "head" in name_lower:
                return LayerType.OUTPUT
            return LayerType.FEED_FORWARD
        
        return LayerType.UNKNOWN
    
    def _compute_criticality(
        self,
        name: str,
        module: nn.Module,
        layer_type: LayerType,
    ) -> float:
        """Compute criticality score for a layer (0.0-1.0)."""
        score = 0.5  # Base score
        
        # Layer type contribution
        type_scores = {
            LayerType.OUTPUT: 0.3,
            LayerType.EMBEDDING: 0.25,
            LayerType.NORM: 0.2,
            LayerType.ATTENTION: 0.1,
            LayerType.FEED_FORWARD: -0.1,
            LayerType.UNKNOWN: 0.0,
        }
        score += type_scores.get(layer_type, 0.0)
        
        # Name pattern contribution
        if any(p in name.lower() for p in self.CRITICAL_PATTERNS):
            score += 0.2
        if any(p in name.lower() for p in self.SAFE_PATTERNS):
            score -= 0.1
        
        # Position in model (first/last layers more critical)
        # This is a heuristic - actual position would need model structure
        if "0" in name or "first" in name.lower():
            score += 0.1
        if "final" in name.lower() or "last" in name.lower():
            score += 0.15
        
        return max(0.0, min(1.0, score))
    
    def _recommend_precision(
        self,
        layer_type: LayerType,
        criticality: float,
        base_precision: PrecisionLevel,
    ) -> PrecisionLevel:
        """Recommend precision based on layer characteristics."""
        
        # Critical layers maintain higher precision
        if layer_type in self._critical_types or criticality > 0.7:
            return base_precision  # Keep original
        
        # High criticality: one step down max
        if criticality > 0.5:
            if base_precision == PrecisionLevel.FP32:
                return PrecisionLevel.FP16
            return base_precision
        
        # Medium criticality: INT8 ok
        if criticality > 0.3:
            return PrecisionLevel.INT8
        
        # Low criticality: INT4 ok
        return PrecisionLevel.INT4
    
    def _get_precision_reasons(
        self,
        layer_type: LayerType,
        criticality: float,
        precision: PrecisionLevel,
    ) -> List[str]:
        """Generate human-readable reasons for precision choice."""
        reasons = []
        
        if layer_type in self._critical_types:
            reasons.append(f"Critical layer type: {layer_type.value}")
        
        if criticality > 0.7:
            reasons.append(f"High criticality score: {criticality:.2f}")
        elif criticality < 0.3:
            reasons.append(f"Low criticality: safe for quantization")
        
        if precision in (PrecisionLevel.INT8, PrecisionLevel.INT4):
            reasons.append("Feed-forward layer: tolerant to quantization")
        
        return reasons
    
    def _estimate_quality_retention(
        self,
        layer_plans: List[LayerCriticality],
    ) -> float:
        """Estimate overall quality retention from precision choices."""
        if not layer_plans:
            return 1.0
        
        total_params = sum(lp.parameter_count for lp in layer_plans)
        if total_params == 0:
            return 1.0
        
        weighted_quality = 0.0
        for lp in layer_plans:
            weight = lp.parameter_count / total_params
            # Critical layers weighted more
            effective_quality = (
                lp.recommended_precision.relative_quality ** 
                (1.0 + lp.criticality_score)
            )
            weighted_quality += weight * effective_quality
        
        return weighted_quality
    
    def _apply_layer_precision(
        self,
        module: nn.Module,
        precision: PrecisionLevel,
        device: str,
    ) -> None:
        """Apply precision to a single layer."""
        
        if precision == PrecisionLevel.FP32:
            module.float()
        elif precision == PrecisionLevel.FP16:
            module.half()
        elif precision == PrecisionLevel.BF16:
            module.to(torch.bfloat16)
        elif precision in (PrecisionLevel.INT8, PrecisionLevel.INT4):
            # Use bitsandbytes if available, otherwise stay at FP16
            try:
                from model.quantizer import quantize_module
                mode = "8bit" if precision == PrecisionLevel.INT8 else "4bit"
                quantize_module(module, mode=mode, device=device)
            except (ImportError, RuntimeError) as e:
                logger.warning(
                    "Quantization not available, falling back to FP16: %s", e
                )
                module.half()


# Register as plugin
@PluginRegistry.register(
    "optimizer", 
    "adaptive_precision",
    description="Dynamic precision adjustment based on layer criticality"
)
class AdaptivePrecisionPlugin(OptimizerPlugin):
    """Plugin wrapper for AdaptivePrecisionController."""
    
    def __init__(
        self,
        memory_threshold: float = 0.85,
        power_threshold: float = 0.80,
        min_quality: float = 0.98,
    ):
        self._controller = AdaptivePrecisionController(
            memory_threshold=memory_threshold,
            power_threshold=power_threshold,
            min_quality=min_quality,
        )
    
    @property
    def name(self) -> str:
        return "adaptive_precision"
    
    def optimize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        base_precision = PrecisionLevel(config.get("base_precision", "fp16"))
        device = config.get("device", "cuda:0")
        
        plan = self._controller.analyze_model(model, base_precision)
        return self._controller.apply_precision(model, plan, device)
    
    def get_optimization_config(self) -> Dict[str, Any]:
        return {
            "base_precision": "fp16",
            "device": "cuda:0",
        }
