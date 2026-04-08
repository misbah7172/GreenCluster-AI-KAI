"""
ONNX Converter Module
Converts PyTorch models to ONNX for cross-platform deployment.

Key Features:
- PyTorch to ONNX conversion
- Model optimization for inference
- Dynamic shape support
- Quantization-aware export
- Multi-hardware simulation support
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import tempfile


class ONNXOptimizationLevel(Enum):
    """ONNX optimization levels."""
    NONE = 0
    BASIC = 1  # Constant folding, dead code elimination
    EXTENDED = 2  # + Operator fusion
    FULL = 3  # + Layout optimization, memory planning


class TargetDevice(Enum):
    """Target device types for optimization."""
    CPU = "cpu"
    CUDA = "cuda"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    COREML = "coreml"
    WEBGPU = "webgpu"


@dataclass
class ExportConfig:
    """Configuration for ONNX export."""
    opset_version: int = 17
    dynamic_axes: Dict[str, Dict[int, str]] = field(default_factory=dict)
    input_names: List[str] = field(default_factory=lambda: ["input_ids"])
    output_names: List[str] = field(default_factory=lambda: ["logits"])
    
    # Optimization settings
    optimization_level: ONNXOptimizationLevel = ONNXOptimizationLevel.EXTENDED
    target_device: TargetDevice = TargetDevice.CPU
    
    # Quantization
    quantize: bool = False
    quantization_type: str = "dynamic"  # dynamic, static
    
    # Export options
    do_constant_folding: bool = True
    export_params: bool = True
    verbose: bool = False
    
    # Model specific
    batch_size: int = 1
    sequence_length: int = 128


@dataclass
class ExportResult:
    """Result of ONNX export."""
    success: bool
    output_path: Optional[str] = None
    model_size_mb: float = 0.0
    export_time_s: float = 0.0
    optimization_time_s: float = 0.0
    original_size_mb: float = 0.0
    compression_ratio: float = 1.0
    
    # Validation results
    validated: bool = False
    max_diff: float = 0.0
    
    # Error information
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Result of ONNX inference."""
    outputs: Dict[str, Any]
    inference_time_ms: float
    memory_usage_mb: float


class ONNXConverter:
    """
    Converts PyTorch models to ONNX format.
    
    Features:
    - Automatic input shape inference
    - Dynamic axes for variable batch/sequence
    - Model optimization and quantization
    - Cross-validation with original model
    """
    
    def __init__(
        self,
        config: Optional[ExportConfig] = None,
    ):
        """
        Initialize ONNX converter.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
        self._has_onnx = self._check_onnx_available()
        self._has_onnxruntime = self._check_onnxruntime_available()
    
    def _check_onnx_available(self) -> bool:
        """Check if ONNX is available."""
        try:
            import onnx
            return True
        except ImportError:
            return False
    
    def _check_onnxruntime_available(self) -> bool:
        """Check if ONNX Runtime is available."""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False
    
    def export(
        self,
        model: Any,  # PyTorch model
        output_path: str,
        sample_input: Optional[Any] = None,
        config: Optional[ExportConfig] = None,
    ) -> ExportResult:
        """
        Export PyTorch model to ONNX.
        
        Args:
            model: PyTorch model to export
            output_path: Path to save ONNX model
            sample_input: Sample input for tracing
            config: Export configuration (overrides default)
            
        Returns:
            ExportResult with status and metrics
        """
        config = config or self.config
        result = ExportResult(success=False)
        
        if not self._has_onnx:
            result.error = "ONNX not installed. Run: pip install onnx"
            return result
        
        try:
            import torch
            import onnx
            
            start_time = time.time()
            
            # Prepare model
            model.eval()
            
            # Create sample input if not provided
            if sample_input is None:
                sample_input = self._create_sample_input(model, config)
            
            # Calculate original size
            original_size = self._get_model_size(model)
            result.original_size_mb = original_size / (1024 * 1024)
            
            # Setup dynamic axes
            dynamic_axes = config.dynamic_axes
            if not dynamic_axes:
                dynamic_axes = self._get_default_dynamic_axes(config)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                sample_input,
                output_path,
                export_params=config.export_params,
                opset_version=config.opset_version,
                do_constant_folding=config.do_constant_folding,
                input_names=config.input_names,
                output_names=config.output_names,
                dynamic_axes=dynamic_axes,
                verbose=config.verbose,
            )
            
            export_time = time.time() - start_time
            result.export_time_s = export_time
            
            # Load and validate ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            # Optimize if requested
            if config.optimization_level != ONNXOptimizationLevel.NONE:
                opt_start = time.time()
                self._optimize_model(output_path, config)
                result.optimization_time_s = time.time() - opt_start
            
            # Quantize if requested
            if config.quantize:
                self._quantize_model(output_path, config)
            
            # Calculate final size
            if os.path.exists(output_path):
                result.model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                result.compression_ratio = result.original_size_mb / result.model_size_mb
            
            result.success = True
            result.output_path = output_path
            
            # Validate output
            if self._has_onnxruntime:
                validation = self._validate_output(model, output_path, sample_input)
                result.validated = validation[0]
                result.max_diff = validation[1]
            
        except Exception as e:
            result.error = str(e)
            result.success = False
        
        return result
    
    def _create_sample_input(self, model: Any, config: ExportConfig) -> Any:
        """Create sample input for model tracing."""
        import torch
        
        # Default: create input_ids tensor
        sample = torch.randint(
            0, 1000,
            (config.batch_size, config.sequence_length),
            dtype=torch.long
        )
        
        # Check if model expects tuple or dict
        # Try to infer from model's forward signature
        try:
            import inspect
            sig = inspect.signature(model.forward)
            params = list(sig.parameters.keys())
            
            if len(params) > 1 and 'attention_mask' in params:
                attention_mask = torch.ones(
                    config.batch_size, config.sequence_length,
                    dtype=torch.long
                )
                return (sample, attention_mask)
        except Exception:
            pass
        
        return sample
    
    def _get_default_dynamic_axes(self, config: ExportConfig) -> Dict:
        """Get default dynamic axes configuration."""
        dynamic_axes = {}
        
        for name in config.input_names:
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        
        for name in config.output_names:
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        
        return dynamic_axes
    
    def _get_model_size(self, model: Any) -> int:
        """Get model size in bytes."""
        import torch
        
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def _optimize_model(self, model_path: str, config: ExportConfig) -> None:
        """Optimize ONNX model."""
        if not self._has_onnxruntime:
            return
        
        try:
            import onnxruntime as ort
            from onnxruntime.transformers import optimizer
            
            # Use ONNX Runtime optimizer
            optimized_model = optimizer.optimize_model(
                model_path,
                model_type='bert',  # Generic transformer type
                num_heads=0,  # Auto-detect
                hidden_size=0,  # Auto-detect
            )
            
            optimized_model.save_model_to_file(model_path)
            
        except ImportError:
            # Try basic ONNX optimization
            try:
                import onnx
                from onnx import optimizer as onnx_optimizer
                
                model = onnx.load(model_path)
                
                # Basic optimizations
                passes = ['eliminate_identity', 'eliminate_deadend', 'fuse_consecutive_transposes']
                
                if config.optimization_level.value >= ONNXOptimizationLevel.EXTENDED.value:
                    passes.extend(['fuse_matmul_add_bias_into_gemm', 'fuse_bn_into_conv'])
                
                optimized = onnx_optimizer.optimize(model, passes)
                onnx.save(optimized, model_path)
                
            except Exception:
                pass  # Skip optimization if it fails
    
    def _quantize_model(self, model_path: str, config: ExportConfig) -> None:
        """Quantize ONNX model."""
        if not self._has_onnxruntime:
            return
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_path = model_path.replace('.onnx', '_quantized.onnx')
            
            quantize_dynamic(
                model_path,
                quantized_path,
                weight_type=QuantType.QInt8,
            )
            
            # Replace original with quantized
            os.replace(quantized_path, model_path)
            
        except ImportError:
            pass
    
    def _validate_output(
        self,
        pytorch_model: Any,
        onnx_path: str,
        sample_input: Any,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> Tuple[bool, float]:
        """
        Validate ONNX output matches PyTorch output.
        
        Returns (is_valid, max_difference).
        """
        import torch
        import onnxruntime as ort
        import numpy as np
        
        try:
            # Get PyTorch output
            pytorch_model.eval()
            with torch.no_grad():
                if isinstance(sample_input, tuple):
                    pytorch_output = pytorch_model(*sample_input)
                else:
                    pytorch_output = pytorch_model(sample_input)
            
            if isinstance(pytorch_output, tuple):
                pytorch_output = pytorch_output[0]
            
            pytorch_np = pytorch_output.numpy()
            
            # Get ONNX output
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            if isinstance(sample_input, tuple):
                onnx_input = sample_input[0].numpy()
            else:
                onnx_input = sample_input.numpy()
            
            onnx_output = session.run(None, {input_name: onnx_input})[0]
            
            # Compare
            max_diff = np.max(np.abs(pytorch_np - onnx_output))
            is_close = np.allclose(pytorch_np, onnx_output, rtol=rtol, atol=atol)
            
            return is_close, float(max_diff)
            
        except Exception as e:
            return False, float('inf')


class ONNXInferenceEngine:
    """
    ONNX inference engine for cross-platform deployment.
    
    Supports multiple execution providers:
    - CPU
    - CUDA
    - TensorRT
    - OpenVINO
    """
    
    def __init__(
        self,
        model_path: str,
        device: TargetDevice = TargetDevice.CPU,
        num_threads: int = 4,
    ):
        """
        Initialize ONNX inference engine.
        
        Args:
            model_path: Path to ONNX model
            device: Target device for inference
            num_threads: Number of threads for CPU inference
        """
        self.model_path = model_path
        self.device = device
        self.num_threads = num_threads
        
        self.session: Optional[Any] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            
            # Set execution providers based on device
            providers = self._get_providers()
            
            # Session options
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.num_threads
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers,
            )
            
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Run: pip install onnxruntime")
    
    def _get_providers(self) -> List[str]:
        """Get execution providers for target device."""
        if self.device == TargetDevice.CUDA:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.device == TargetDevice.TENSORRT:
            return ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.device == TargetDevice.OPENVINO:
            return ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']
    
    def infer(
        self,
        inputs: Dict[str, Any],
        output_names: Optional[List[str]] = None,
    ) -> InferenceResult:
        """
        Run inference on input.
        
        Args:
            inputs: Dictionary of input name -> numpy array
            output_names: Specific outputs to return (None = all)
            
        Returns:
            InferenceResult with outputs and timing
        """
        import numpy as np
        
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Run inference
        outputs = self.session.run(
            output_names or self.output_names,
            inputs
        )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Create output dict
        output_dict = {}
        names = output_names or self.output_names
        for i, name in enumerate(names):
            output_dict[name] = outputs[i]
        
        # Estimate memory usage
        memory_usage = sum(
            arr.nbytes for arr in outputs if isinstance(arr, np.ndarray)
        ) / (1024 * 1024)  # MB
        
        return InferenceResult(
            outputs=output_dict,
            inference_time_ms=inference_time,
            memory_usage_mb=memory_usage,
        )
    
    def get_input_info(self) -> List[Dict[str, Any]]:
        """Get information about model inputs."""
        if self.session is None:
            return []
        
        info = []
        for inp in self.session.get_inputs():
            info.append({
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type,
            })
        return info
    
    def get_output_info(self) -> List[Dict[str, Any]]:
        """Get information about model outputs."""
        if self.session is None:
            return []
        
        info = []
        for out in self.session.get_outputs():
            info.append({
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
            })
        return info
    
    def benchmark(
        self,
        sample_input: Dict[str, Any],
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Returns timing statistics.
        """
        import numpy as np
        
        # Warmup
        for _ in range(warmup_runs):
            self.infer(sample_input)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            result = self.infer(sample_input)
            times.append(result.inference_time_ms)
        
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "throughput_per_sec": 1000.0 / float(np.mean(times)),
        }


class ModelSimulator:
    """
    Simulates model execution across different hardware platforms.
    Uses ONNX for cross-platform compatibility.
    """
    
    def __init__(self):
        self.converter = ONNXConverter()
        self.engines: Dict[str, ONNXInferenceEngine] = {}
    
    def prepare_model(
        self,
        pytorch_model: Any,
        model_name: str,
        targets: List[TargetDevice] = None,
    ) -> Dict[str, str]:
        """
        Prepare model for simulation on multiple targets.
        
        Returns dict of target -> onnx_path.
        """
        targets = targets or [TargetDevice.CPU]
        
        onnx_paths = {}
        
        for target in targets:
            # Export with target-specific optimizations
            config = ExportConfig(
                target_device=target,
                optimization_level=ONNXOptimizationLevel.EXTENDED,
            )
            
            output_path = f"{model_name}_{target.value}.onnx"
            result = self.converter.export(pytorch_model, output_path, config=config)
            
            if result.success:
                onnx_paths[target.value] = output_path
                
                # Create inference engine
                try:
                    self.engines[target.value] = ONNXInferenceEngine(
                        output_path,
                        device=target,
                    )
                except Exception:
                    pass  # Provider not available
        
        return onnx_paths
    
    def simulate(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, InferenceResult]:
        """
        Simulate inference on specified targets.
        
        Returns results per target.
        """
        targets = targets or list(self.engines.keys())
        results = {}
        
        for target in targets:
            if target in self.engines:
                try:
                    results[target] = self.engines[target].infer(inputs)
                except Exception as e:
                    results[target] = InferenceResult(
                        outputs={},
                        inference_time_ms=-1,
                        memory_usage_mb=0,
                    )
        
        return results
    
    def compare_performance(
        self,
        inputs: Dict[str, Any],
        num_runs: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance across all loaded targets.
        
        Returns benchmark results per target.
        """
        results = {}
        
        for target, engine in self.engines.items():
            try:
                results[target] = engine.benchmark(inputs, num_runs)
            except Exception as e:
                results[target] = {"error": str(e)}
        
        return results


# Plugin registration
try:
    from model.plugin_architecture import PluginRegistry
    
    @PluginRegistry.register("converter", "onnx")
    class ONNXConverterPlugin:
        """ONNX Converter Plugin."""
        
        def __init__(self, **kwargs):
            self.converter = ONNXConverter(**kwargs)
        
        def export(self, model, output_path, **kwargs) -> ExportResult:
            return self.converter.export(model, output_path, **kwargs)

except ImportError:
    pass
