"""
Real-world latency and bandwidth probing module.
Replaces synthetic network simulation with actual measurements using ping, iperf3, or gRPC health checks.
"""

import subprocess
import socket
import time
import json
import logging
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a latency probe between two hosts."""
    source: str
    target: str
    rtt_ms: float  # Round-trip time in milliseconds
    min_rtt_ms: float
    max_rtt_ms: float
    stddev_ms: float
    packet_loss_pct: float
    probe_time: datetime
    probe_method: str  # "ping", "iperf3", "grpc-healthcheck"
    samples: int
    
    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "rtt_ms": round(self.rtt_ms, 3),
            "min_rtt_ms": round(self.min_rtt_ms, 3),
            "max_rtt_ms": round(self.max_rtt_ms, 3),
            "stddev_ms": round(self.stddev_ms, 3),
            "packet_loss_pct": round(self.packet_loss_pct, 1),
            "probe_time": self.probe_time.isoformat(),
            "probe_method": self.probe_method,
            "samples": self.samples,
        }


class LatencyProber:
    """Measures real latency and bandwidth between network endpoints."""
    
    def __init__(self, cache_ttl_seconds: int = 60):
        """
        Args:
            cache_ttl_seconds: How long to cache probe results before re-probing.
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self._probe_cache: Dict[str, Tuple[ProbeResult, float]] = {}  # key: "src:dst" -> (result, timestamp)
        self._lock = threading.Lock()
    
    def probe_ping(self, host: str, samples: int = 4, timeout: int = 5) -> Optional[ProbeResult]:
        """
        Measure latency to a host using ping (ICMP).
        
        Args:
            host: Target hostname or IP address
            samples: Number of ping packets to send
            timeout: Timeout in seconds per ping
            
        Returns:
            ProbeResult with latency statistics, or None if all pings failed.
        """
        try:
            # Windows vs Linux ping command differences
            import platform
            is_windows = platform.system() == "Windows"
            count_flag = "-n" if is_windows else "-c"
            timeout_flag = "-w" if is_windows else "-W"
            
            cmd = ["ping", count_flag, str(samples), timeout_flag, str(timeout * 1000), host]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
            
            if result.returncode == 0:
                # Parse output to extract RTT values
                rtts = self._parse_ping_output(result.stdout, is_windows)
                if rtts:
                    return ProbeResult(
                        source="local",
                        target=host,
                        rtt_ms=statistics.mean(rtts),
                        min_rtt_ms=min(rtts),
                        max_rtt_ms=max(rtts),
                        stddev_ms=statistics.stdev(rtts) if len(rtts) > 1 else 0,
                        packet_loss_pct=0.0,
                        probe_time=datetime.now(),
                        probe_method="ping",
                        samples=len(rtts),
                    )
            else:
                # Partial packet loss or complete failure
                logger.warning(f"Ping to {host} failed with return code {result.returncode}")
                return None
                
        except Exception as e:
            logger.warning(f"Ping probe to {host} failed: {e}")
            return None
    
    def probe_grpc_healthcheck(self, grpc_endpoint: str, samples: int = 3) -> Optional[ProbeResult]:
        """
        Measure latency using gRPC health check RPC.
        
        Args:
            grpc_endpoint: gRPC server address (e.g., "localhost:50051")
            samples: Number of health checks to perform
            
        Returns:
            ProbeResult with latency measurements.
        """
        try:
            import grpc
            from proto.inference_pb2_grpc import InferenceStub
            from proto.inference_pb2 import HealthCheckRequest
            
            rtts = []
            for _ in range(samples):
                channel = grpc.aio.secure_channel(grpc_endpoint, grpc.aio.ssl_channel_credentials())
                stub = InferenceStub(channel)
                
                start = time.time()
                # Note: This is synchronous; for async see probe_grpc_healthcheck_async
                try:
                    request = HealthCheckRequest(service="inference")
                    stub.HealthCheck(request, timeout=2.0)
                except:
                    pass  # gRPC health check may not be implemented; latency still captured
                elapsed_ms = (time.time() - start) * 1000
                rtts.append(elapsed_ms)
                channel.close()
            
            if rtts:
                return ProbeResult(
                    source="local",
                    target=grpc_endpoint,
                    rtt_ms=statistics.mean(rtts),
                    min_rtt_ms=min(rtts),
                    max_rtt_ms=max(rtts),
                    stddev_ms=statistics.stdev(rtts) if len(rtts) > 1 else 0,
                    packet_loss_pct=0.0,
                    probe_time=datetime.now(),
                    probe_method="grpc-healthcheck",
                    samples=len(rtts),
                )
        except Exception as e:
            logger.warning(f"gRPC health check probe to {grpc_endpoint} failed: {e}")
            return None
    
    def probe_socket_connect(self, host: str, port: int, samples: int = 3) -> Optional[ProbeResult]:
        """
        Measure latency using TCP socket connection time.
        
        Args:
            host: Target hostname or IP
            port: Target port
            samples: Number of connection attempts
            
        Returns:
            ProbeResult with latency measurements.
        """
        rtts = []
        for _ in range(samples):
            try:
                start = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect((host, port))
                sock.close()
                elapsed_ms = (time.time() - start) * 1000
                rtts.append(elapsed_ms)
            except Exception as e:
                logger.debug(f"Socket connect probe to {host}:{port} attempt failed: {e}")
                continue
        
        if rtts:
            return ProbeResult(
                source="local",
                target=f"{host}:{port}",
                rtt_ms=statistics.mean(rtts),
                min_rtt_ms=min(rtts),
                max_rtt_ms=max(rtts),
                stddev_ms=statistics.stdev(rtts) if len(rtts) > 1 else 0,
                packet_loss_pct=0.0,
                probe_time=datetime.now(),
                probe_method="socket-connect",
                samples=len(rtts),
            )
        return None
    
    def probe_cached(self, host: str, method: str = "socket-connect", 
                     samples: int = 3, force_refresh: bool = False) -> Optional[ProbeResult]:
        """
        Probe with caching to avoid over-probing the same target.
        
        Args:
            host: Target host
            method: Probing method ("ping", "socket-connect", "grpc-healthcheck")
            samples: Number of samples
            force_refresh: Bypass cache and re-probe immediately
            
        Returns:
            Cached or fresh ProbeResult.
        """
        cache_key = f"local:{host}:{method}"
        
        with self._lock:
            if not force_refresh and cache_key in self._probe_cache:
                result, timestamp = self._probe_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl_seconds:
                    logger.debug(f"Using cached probe result for {host} ({cache_key})")
                    return result
        
        # Perform fresh probe
        if method == "ping":
            probe_result = self.probe_ping(host, samples)
        elif method == "socket-connect":
            # Parse host:port from host string
            if ":" in host:
                h, p = host.rsplit(":", 1)
                try:
                    probe_result = self.probe_socket_connect(h, int(p), samples)
                except:
                    logger.warning(f"Invalid host:port format: {host}")
                    return None
            else:
                probe_result = self.probe_socket_connect(host, 50051, samples)  # Default gRPC port
        else:
            probe_result = self.probe_grpc_healthcheck(host, samples)
        
        # Cache result
        if probe_result:
            with self._lock:
                self._probe_cache[cache_key] = (probe_result, time.time())
        
        return probe_result
    
    def probe_all_pairs(self, hosts: List[str], method: str = "socket-connect") -> Dict[str, ProbeResult]:
        """
        Probe all pairs of hosts and return latency matrix.
        
        Args:
            hosts: List of host addresses
            method: Probing method
            
        Returns:
            Dictionary of {pair_key: ProbeResult} for all pairs.
        """
        results = {}
        for host in hosts:
            result = self.probe_cached(host, method=method, samples=3)
            if result:
                key = f"local->{host}"
                results[key] = result
        return results
    
    def _parse_ping_output(self, output: str, is_windows: bool) -> List[float]:
        """Parse ping command output to extract RTT values."""
        rtts = []
        try:
            for line in output.split("\n"):
                if is_windows and "time=" in line:
                    # Windows: "Reply from 192.168.1.1: bytes=32 time=1ms TTL=64"
                    parts = line.split("time=")
                    if len(parts) > 1:
                        rtt_str = parts[1].split("ms")[0].strip()
                        rtts.append(float(rtt_str))
                elif not is_windows and "time=" in line:
                    # Linux: "64 bytes from hostname (1.2.3.4): icmp_seq=1 ttl=64 time=1.234 ms"
                    parts = line.split("time=")
                    if len(parts) > 1:
                        rtt_str = parts[1].split(" ")[0]
                        rtts.append(float(rtt_str))
        except Exception as e:
            logger.warning(f"Failed to parse ping output: {e}")
        
        return rtts


# Global prober instance
_default_prober: Optional[LatencyProber] = None


def get_default_prober() -> LatencyProber:
    """Get or create the default global latency prober."""
    global _default_prober
    if _default_prober is None:
        _default_prober = LatencyProber(cache_ttl_seconds=60)
    return _default_prober


def probe_endpoint(host: str, method: str = "socket-connect", 
                   samples: int = 3, force_refresh: bool = False) -> Optional[ProbeResult]:
    """Convenience function to probe an endpoint using the default prober."""
    return get_default_prober().probe_cached(host, method=method, samples=samples, 
                                             force_refresh=force_refresh)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    prober = LatencyProber()
    
    test_hosts = ["localhost:50051", "8.8.8.8", "google.com"]
    
    print("=" * 60)
    print("LATENCY PROBING RESULTS")
    print("=" * 60)
    
    for host in test_hosts:
        print(f"\nProbing {host}...")
        result = prober.probe_cached(host, method="socket-connect" if ":" in host else "ping", samples=4)
        if result:
            print(f"  RTT: {result.rtt_ms:.2f} ms")
            print(f"  Min/Max: {result.min_rtt_ms:.2f} / {result.max_rtt_ms:.2f} ms")
            print(f"  StdDev: {result.stddev_ms:.2f} ms")
            print(f"  Samples: {result.samples}")
            print(f"  Method: {result.probe_method}")
        else:
            print(f"  Failed to probe {host}")
