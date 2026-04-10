"""
Telemetry and performance metrics collection system.
Tracks routing decisions, latencies, throughput, and resource utilization for dashboards and analysis.
"""

import json
import time
import threading
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Record of a single routing decision."""
    timestamp: float  # Unix epoch
    chunk_index: int
    selected_host: str
    candidate_hosts: List[str]
    latencies_ms: Dict[str, float]  # host -> latency
    decision_latency_ms: float  # Time taken to decide
    inference_id: str  # Unique inference session ID
    model_name: str
    method: str  # "deterministic", "random", "ml-based", etc.


@dataclass
class InferenceMetrics:
    """Performance metrics for a complete inference."""
    inference_id: str
    model_name: str
    start_time: float
    end_time: Optional[float] = None
    total_duration_ms: Optional[float] = None
    chunk_count: int = 0
    total_tokens: int = 0
    tokens_per_second: Optional[float] = None
    gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.end_time and self.start_time:
            self.total_duration_ms = (self.end_time - self.start_time) * 1000
            if self.total_tokens > 0:
                self.tokens_per_second = self.total_tokens / ((self.end_time - self.start_time) + 0.001)


class TelemetryCollector:
    """Central telemetry collection point for performance metrics."""
    
    def __init__(self, max_history: int = 10000):
        """
        Args:
            max_history: Maximum number of historical events to retain.
        """
        self.max_history = max_history
        self._routing_decisions: deque = deque(maxlen=max_history)
        self._inferences: deque = deque(maxlen=max_history)
        self._latency_samples: defaultdict = defaultdict(deque)  # host -> deque of (timestamp, rtt_ms)
        self._throughput_samples: deque = deque(maxlen=max_history)  # (timestamp, tokens/sec)
        self._lock = threading.RLock()
        self._session_start = time.time()
        
        # Aggregated stats
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_time = 0
        self._stats_cache_ttl = 5  # Recompute stats every 5 seconds
    
    def record_routing_decision(self, decision: RoutingDecision) -> None:
        """Record a routing decision."""
        with self._lock:
            self._routing_decisions.append(decision)
            logger.debug(f"Recorded routing decision: chunk {decision.chunk_index} -> {decision.selected_host}")
    
    def record_inference(self, metrics: InferenceMetrics) -> None:
        """Record inference metrics."""
        with self._lock:
            self._inferences.append(metrics)
            if metrics.tokens_per_second and metrics.tokens_per_second > 0:
                self._throughput_samples.append((time.time(), metrics.tokens_per_second))
            logger.debug(f"Recorded inference: {metrics.inference_id} ({metrics.model_name}) {metrics.total_duration_ms:.1f}ms")
    
    def record_latency_sample(self, host: str, rtt_ms: float) -> None:
        """Record a latency measurement for a host."""
        with self._lock:
            self._latency_samples[host].append((time.time(), rtt_ms))
            # Keep only last 1000 samples per host
            while len(self._latency_samples[host]) > 1000:
                self._latency_samples[host].popleft()
    
    def get_routing_statistics(self, time_window_seconds: int = 300) -> Dict[str, Any]:
        """
        Get routing decision statistics within a time window.
        
        Args:
            time_window_seconds: Look at decisions from last N seconds
            
        Returns:
            Dictionary with routing statistics.
        """
        with self._lock:
            now = time.time()
            cutoff = now - time_window_seconds
            
            recent_decisions = [d for d in self._routing_decisions 
                              if d.timestamp >= cutoff]
            
            if not recent_decisions:
                return {"total_decisions": 0, "hosts": {}}
            
            host_selection_count = defaultdict(int)
            host_avg_latencies = defaultdict(list)
            decision_times = []
            
            for decision in recent_decisions:
                host_selection_count[decision.selected_host] += 1
                decision_times.append(decision.decision_latency_ms)
                for host, lat in decision.latencies_ms.items():
                    host_avg_latencies[host].append(lat)
            
            return {
                "total_decisions": len(recent_decisions),
                "time_window_seconds": time_window_seconds,
                "avg_decision_latency_ms": statistics.mean(decision_times) if decision_times else 0,
                "hosts": {
                    host: {
                        "selection_count": host_selection_count[host],
                        "selection_rate_pct": 100 * host_selection_count[host] / len(recent_decisions),
                        "avg_observed_latency_ms": statistics.mean(host_avg_latencies[host]) if host_avg_latencies[host] else 0,
                        "min_latency_ms": min(host_avg_latencies[host]) if host_avg_latencies[host] else 0,
                        "max_latency_ms": max(host_avg_latencies[host]) if host_avg_latencies[host] else 0,
                    }
                    for host in set(host_selection_count.keys()) | set(host_avg_latencies.keys())
                }
            }
    
    def get_inference_statistics(self, time_window_seconds: int = 300) -> Dict[str, Any]:
        """
        Get inference performance statistics within a time window.
        
        Args:
            time_window_seconds: Look at inferences from last N seconds
            
        Returns:
            Dictionary with inference statistics.
        """
        with self._lock:
            now = time.time()
            cutoff = now - time_window_seconds
            
            recent_inferences = [i for i in self._inferences 
                               if i.start_time >= cutoff and i.end_time]
            
            if not recent_inferences:
                return {"total_inferences": 0}
            
            durations = [i.total_duration_ms for i in recent_inferences if i.total_duration_ms]
            throughputs = [i.tokens_per_second for i in recent_inferences if i.tokens_per_second and i.tokens_per_second > 0]
            
            return {
                "total_inferences": len(recent_inferences),
                "time_window_seconds": time_window_seconds,
                "avg_duration_ms": statistics.mean(durations) if durations else 0,
                "min_duration_ms": min(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0,
                "median_duration_ms": statistics.median(durations) if durations else 0,
                "avg_tokens_per_second": statistics.mean(throughputs) if throughputs else 0,
                "max_tokens_per_second": max(throughputs) if throughputs else 0,
                "total_chunks_processed": sum(i.chunk_count for i in recent_inferences),
                "success_rate_pct": 100 * len([i for i in recent_inferences if not i.errors]) / len(recent_inferences),
            }
    
    def get_latency_statistics(self, host: str, time_window_seconds: int = 300) -> Dict[str, Any]:
        """
        Get latency statistics for a specific host.
        
        Args:
            host: Target host
            time_window_seconds: Time window
            
        Returns:
            Dictionary with latency statistics.
        """
        with self._lock:
            now = time.time()
            cutoff = now - time_window_seconds
            
            samples = [rtt for ts, rtt in self._latency_samples[host] if ts >= cutoff]
            
            if not samples:
                return {"host": host, "samples_count": 0}
            
            return {
                "host": host,
                "samples_count": len(samples),
                "avg_latency_ms": statistics.mean(samples),
                "min_latency_ms": min(samples),
                "max_latency_ms": max(samples),
                "median_latency_ms": statistics.median(samples),
                "stddev_ms": statistics.stdev(samples) if len(samples) > 1 else 0,
                "p95_latency_ms": sorted(samples)[int(0.95 * len(samples))] if samples else 0,
                "p99_latency_ms": sorted(samples)[int(0.99 * len(samples))] if samples else 0,
            }
    
    def get_throughput_statistics(self, time_window_seconds: int = 300) -> Dict[str, Any]:
        """Get overall throughput statistics."""
        with self._lock:
            now = time.time()
            cutoff = now - time_window_seconds
            
            samples = [tps for ts, tps in self._throughput_samples if ts >= cutoff]
            
            if not samples:
                return {"samples_count": 0, "avg_tokens_per_second": 0}
            
            return {
                "samples_count": len(samples),
                "avg_tokens_per_second": statistics.mean(samples),
                "max_tokens_per_second": max(samples),
                "min_tokens_per_second": min(samples),
                "median_tokens_per_second": statistics.median(samples),
            }
    
    def get_summary(self, time_window_seconds: int = 300) -> Dict[str, Any]:
        """Get comprehensive summary of all metrics."""
        with self._lock:
            now = time.time()
            
            # Check cache
            if now - self._stats_cache_time < self._stats_cache_ttl and self._stats_cache:
                return self._stats_cache
        
        routing_stats = self.get_routing_statistics(time_window_seconds)
        inference_stats = self.get_inference_statistics(time_window_seconds)
        throughput_stats = self.get_throughput_statistics(time_window_seconds)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self._session_start,
            "time_window_seconds": time_window_seconds,
            "routing": routing_stats,
            "inference": inference_stats,
            "throughput": throughput_stats,
            "latency_by_host": {
                host: self.get_latency_statistics(host, time_window_seconds)
                for host in list(self._latency_samples.keys())[:10]  # Limit to 10 hosts in summary
            }
        }
        
        with self._lock:
            self._stats_cache = summary
            self._stats_cache_time = now
        
        return summary
    
    def export_json(self, filepath: str, time_window_seconds: int = 300) -> None:
        """Export metrics summary to JSON file."""
        summary = self.get_summary(time_window_seconds)
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Exported telemetry to {filepath}")
    
    def get_routing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent routing decisions as dicts."""
        with self._lock:
            return [asdict(d) for d in list(self._routing_decisions)[-limit:]]
    
    def get_inference_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent inferences as dicts."""
        with self._lock:
            return [asdict(m) for m in list(self._inferences)[-limit:]]
    
    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._routing_decisions.clear()
            self._inferences.clear()
            self._latency_samples.clear()
            self._throughput_samples.clear()
            self._session_start = time.time()
            logger.info("Telemetry collector reset")


# Global collector instance
_default_collector: Optional[TelemetryCollector] = None


def get_default_collector() -> TelemetryCollector:
    """Get or create the default global telemetry collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = TelemetryCollector(max_history=10000)
    return _default_collector


def record_routing_decision(decision: RoutingDecision) -> None:
    """Convenience function to record routing decision."""
    get_default_collector().record_routing_decision(decision)


def record_inference(metrics: InferenceMetrics) -> None:
    """Convenience function to record inference metrics."""
    get_default_collector().record_inference(metrics)


def record_latency_sample(host: str, rtt_ms: float) -> None:
    """Convenience function to record latency sample."""
    get_default_collector().record_latency_sample(host, rtt_ms)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    collector = TelemetryCollector()
    
    # Simulate routing decisions
    for i in range(50):
        decision = RoutingDecision(
            timestamp=time.time(),
            chunk_index=i % 4,
            selected_host=f"node{i % 3}:50051",
            candidate_hosts=[f"node{j}:50051" for j in range(3)],
            latencies_ms={f"node{j}:50051": 10 + (j + i) % 5 for j in range(3)},
            decision_latency_ms=0.5,
            inference_id=f"inf_{i // 4}",
            model_name="test-model",
            method="deterministic",
        )
        collector.record_routing_decision(decision)
        
        # Simulate latency samples
        for j in range(3):
            collector.record_latency_sample(f"node{j}:50051", 10 + (j + i) % 5)
    
    # Simulate inferences
    for i in range(5):
        start = time.time() - 100 + i * 20
        end = start + 15
        metrics = InferenceMetrics(
            inference_id=f"inf_{i}",
            model_name="test-model",
            start_time=start,
            end_time=end,
            chunk_count=4,
            total_tokens=512,
        )
        collector.record_inference(metrics)
    
    # Print summary
    summary = collector.get_summary(time_window_seconds=600)
    print(json.dumps(summary, indent=2))
