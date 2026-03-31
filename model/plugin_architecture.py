"""
Modular Plugin Architecture for KAI.

Enables pluggable subsystems for scheduler, optimizer, execution engine,
and cache system. Each subsystem can be swapped without modifying core code.

Usage::

    from model.plugin_architecture import PluginRegistry, SchedulerPlugin
    
    # Register a custom scheduler
    @PluginRegistry.register("scheduler", "my_scheduler")
    class MyScheduler(SchedulerPlugin):
        def schedule(self, layers, nodes):
            ...
    
    # Get a scheduler by name
    scheduler = PluginRegistry.get("scheduler", "my_scheduler")
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PluginType(Enum):
    """Categories of pluggable subsystems."""
    SCHEDULER = "scheduler"
    OPTIMIZER = "optimizer"
    EXECUTOR = "executor"
    CACHE = "cache"
    QUANTIZER = "quantizer"
    PLACEMENT = "placement"
    PARALLELISM = "parallelism"


@dataclass
class PluginMetadata:
    """Metadata for registered plugins."""
    name: str
    plugin_type: PluginType
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)


class PluginRegistry:
    """Global registry for KAI plugins.
    
    Thread-safe singleton that maintains registered plugins by type and name.
    """
    
    _plugins: Dict[PluginType, Dict[str, Type]] = {}
    _metadata: Dict[str, PluginMetadata] = {}
    _instances: Dict[str, Any] = {}
    
    @classmethod
    def register(
        cls,
        plugin_type: str,
        name: str,
        version: str = "1.0.0",
        description: str = "",
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a plugin class.
        
        Parameters
        ----------
        plugin_type : str
            Type of plugin (scheduler, optimizer, executor, cache, etc.)
        name : str
            Unique name for this plugin
        version : str
            Version string
        description : str
            Human-readable description
            
        Returns
        -------
        decorator
            Class decorator that registers the plugin
        """
        ptype = PluginType(plugin_type)
        
        def decorator(plugin_class: Type[T]) -> Type[T]:
            if ptype not in cls._plugins:
                cls._plugins[ptype] = {}
            
            cls._plugins[ptype][name] = plugin_class
            cls._metadata[f"{ptype.value}:{name}"] = PluginMetadata(
                name=name,
                plugin_type=ptype,
                version=version,
                description=description,
            )
            
            logger.info("Registered plugin: %s/%s (v%s)", plugin_type, name, version)
            return plugin_class
        
        return decorator
    
    @classmethod
    def get(cls, plugin_type: str, name: str, **kwargs) -> Any:
        """Get an instance of a registered plugin.
        
        Parameters
        ----------
        plugin_type : str
            Type of plugin
        name : str
            Plugin name
        **kwargs
            Arguments passed to plugin constructor
            
        Returns
        -------
        plugin instance
        """
        ptype = PluginType(plugin_type)
        
        if ptype not in cls._plugins or name not in cls._plugins[ptype]:
            raise KeyError(f"Plugin not found: {plugin_type}/{name}")
        
        plugin_class = cls._plugins[ptype][name]
        return plugin_class(**kwargs)
    
    @classmethod
    def get_singleton(cls, plugin_type: str, name: str, **kwargs) -> Any:
        """Get or create a singleton instance of a plugin."""
        key = f"{plugin_type}:{name}"
        
        if key not in cls._instances:
            cls._instances[key] = cls.get(plugin_type, name, **kwargs)
        
        return cls._instances[key]
    
    @classmethod
    def list_plugins(cls, plugin_type: Optional[str] = None) -> List[PluginMetadata]:
        """List all registered plugins, optionally filtered by type."""
        if plugin_type is not None:
            ptype = PluginType(plugin_type)
            return [
                meta for key, meta in cls._metadata.items()
                if meta.plugin_type == ptype
            ]
        return list(cls._metadata.values())
    
    @classmethod
    def has_plugin(cls, plugin_type: str, name: str) -> bool:
        """Check if a plugin is registered."""
        ptype = PluginType(plugin_type)
        return ptype in cls._plugins and name in cls._plugins[ptype]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered plugins (for testing)."""
        cls._plugins.clear()
        cls._metadata.clear()
        cls._instances.clear()


# ---------------------------------------------------------------------------
# Base Plugin Interfaces
# ---------------------------------------------------------------------------

class BasePlugin(ABC):
    """Base class for all KAI plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the plugin name."""
        pass
    
    @property
    def version(self) -> str:
        """Return the plugin version."""
        return "1.0.0"
    
    def initialize(self) -> None:
        """Initialize the plugin (called once)."""
        pass
    
    def shutdown(self) -> None:
        """Cleanup resources."""
        pass


class SchedulerPlugin(BasePlugin):
    """Interface for scheduling plugins."""
    
    @abstractmethod
    def schedule(
        self,
        layers: List[Any],
        nodes: List[Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[int]]:
        """Assign layers to nodes.
        
        Parameters
        ----------
        layers : list
            Model layers to schedule
        nodes : list
            Available cluster nodes
        constraints : dict, optional
            Scheduling constraints (memory, latency, energy)
            
        Returns
        -------
        dict
            Mapping of node_name -> list of layer indices
        """
        pass
    
    def rebalance(
        self,
        current_mapping: Dict[str, List[int]],
        metrics: Dict[str, Any],
    ) -> Optional[Dict[str, List[int]]]:
        """Rebalance layers based on runtime metrics.
        
        Returns None if no rebalancing needed.
        """
        return None


class OptimizerPlugin(BasePlugin):
    """Interface for optimization plugins."""
    
    @abstractmethod
    def optimize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """Apply optimizations to a model.
        
        Parameters
        ----------
        model : nn.Module
            Model to optimize
        config : dict
            Optimization configuration
            
        Returns
        -------
        nn.Module
            Optimized model
        """
        pass
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Return default optimization configuration."""
        return {}


class ExecutorPlugin(BasePlugin):
    """Interface for execution engine plugins."""
    
    @abstractmethod
    def execute(
        self,
        module: nn.Module,
        inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Execute a forward pass.
        
        Parameters
        ----------
        module : nn.Module
            Module to execute
        inputs : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Output tensor
        """
        pass
    
    def supports_async(self) -> bool:
        """Return True if async execution is supported."""
        return False
    
    async def execute_async(
        self,
        module: nn.Module,
        inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Async execution (optional)."""
        return self.execute(module, inputs, **kwargs)


class CachePlugin(BasePlugin):
    """Interface for cache plugins."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value in cache."""
        pass
    
    @abstractmethod
    def evict(self, key: str) -> bool:
        """Remove a cached value."""
        pass
    
    def clear(self) -> None:
        """Clear all cached values."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {}


class PlacementPlugin(BasePlugin):
    """Interface for model placement plugins."""
    
    @abstractmethod
    def compute_placement(
        self,
        layers: List[Any],
        nodes: List[Any],
        constraints: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compute optimal layer-to-node placement.
        
        Parameters
        ----------
        layers : list
            Model layers with size estimates
        nodes : list
            Available nodes with resources
        constraints : dict
            Placement constraints (latency, energy, bandwidth)
            
        Returns
        -------
        list[dict]
            Placement plan for each layer
        """
        pass


class ParallelismPlugin(BasePlugin):
    """Interface for parallelism strategy plugins."""
    
    @abstractmethod
    def get_strategy(
        self,
        layer_type: str,
        resources: Dict[str, Any],
    ) -> str:
        """Determine parallelism strategy for a layer.
        
        Returns
        -------
        str
            Strategy name: "pipeline", "tensor", "hybrid"
        """
        pass
    
    @abstractmethod
    def execute_parallel(
        self,
        module: nn.Module,
        inputs: torch.Tensor,
        strategy: str,
        devices: List[str],
    ) -> torch.Tensor:
        """Execute module with specified parallelism strategy."""
        pass


# ---------------------------------------------------------------------------
# Default Plugin Implementations
# ---------------------------------------------------------------------------

@PluginRegistry.register("scheduler", "default", description="Simple round-robin scheduler")
class DefaultScheduler(SchedulerPlugin):
    """Default round-robin scheduler."""
    
    @property
    def name(self) -> str:
        return "default"
    
    def schedule(
        self,
        layers: List[Any],
        nodes: List[Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[int]]:
        if not nodes:
            raise ValueError("No nodes available for scheduling")
        
        mapping: Dict[str, List[int]] = {n.name if hasattr(n, 'name') else str(i): [] 
                                          for i, n in enumerate(nodes)}
        node_names = list(mapping.keys())
        
        for i, _ in enumerate(layers):
            node = node_names[i % len(node_names)]
            mapping[node].append(i)
        
        return mapping


@PluginRegistry.register("cache", "lru", description="LRU eviction cache")
class LRUCache(CachePlugin):
    """Simple LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Any] = {}
        self._order: List[str] = []
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    @property
    def name(self) -> str:
        return "lru"
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        self._misses += 1
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self._max_size:
            # Evict least recently used
            old_key = self._order.pop(0)
            del self._cache[old_key]
        
        self._cache[key] = value
        self._order.append(key)
    
    def evict(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            self._order.remove(key)
            return True
        return False
    
    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


@PluginRegistry.register("executor", "default", description="Standard PyTorch executor")
class DefaultExecutor(ExecutorPlugin):
    """Default PyTorch forward execution."""
    
    @property
    def name(self) -> str:
        return "default"
    
    def execute(
        self,
        module: nn.Module,
        inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        with torch.no_grad():
            return module(inputs)


# ---------------------------------------------------------------------------
# Plugin Configuration
# ---------------------------------------------------------------------------

@dataclass
class PluginConfig:
    """Configuration for instantiating plugins."""
    
    scheduler: str = "default"
    scheduler_args: Dict[str, Any] = field(default_factory=dict)
    
    optimizer: Optional[str] = None
    optimizer_args: Dict[str, Any] = field(default_factory=dict)
    
    executor: str = "default"
    executor_args: Dict[str, Any] = field(default_factory=dict)
    
    cache: str = "lru"
    cache_args: Dict[str, Any] = field(default_factory=lambda: {"max_size": 1000})
    
    def create_scheduler(self) -> SchedulerPlugin:
        return PluginRegistry.get("scheduler", self.scheduler, **self.scheduler_args)
    
    def create_executor(self) -> ExecutorPlugin:
        return PluginRegistry.get("executor", self.executor, **self.executor_args)
    
    def create_cache(self) -> CachePlugin:
        return PluginRegistry.get("cache", self.cache, **self.cache_args)


def load_plugins_from_directory(plugin_dir: str) -> int:
    """Load plugin modules from a directory.
    
    Each .py file in the directory is imported, which triggers
    @PluginRegistry.register decorators.
    
    Returns the number of plugins loaded.
    """
    import importlib.util
    import os
    
    count = 0
    
    if not os.path.isdir(plugin_dir):
        return count
    
    for filename in os.listdir(plugin_dir):
        if filename.endswith(".py") and not filename.startswith("_"):
            filepath = os.path.join(plugin_dir, filename)
            module_name = f"kai_plugin_{filename[:-3]}"
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    count += 1
                    logger.info("Loaded plugin module: %s", filename)
            except Exception as e:
                logger.error("Failed to load plugin %s: %s", filename, e)
    
    return count
