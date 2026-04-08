"""
Batch Processor Module
Processes multiple inference requests together for improved efficiency.

Key Features:
- Dynamic batching with configurable window
- Request queue management
- Padding and sequence handling
- Batch-aware KV cache management
- Priority-based batch formation
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from queue import PriorityQueue
from enum import Enum
import math


class BatchingStrategy(Enum):
    """Batching strategy types."""
    FIXED_SIZE = "fixed_size"  # Wait for fixed batch size
    FIXED_TIME = "fixed_time"  # Wait for fixed time window
    ADAPTIVE = "adaptive"  # Dynamically adjust based on load
    CONTINUOUS = "continuous"  # Continuous batching (iteration-level)


class RequestStatus(Enum):
    """Status of an inference request."""
    QUEUED = "queued"
    BATCHED = "batched"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class InferenceRequest:
    """Single inference request."""
    request_id: str
    prompt: str
    max_tokens: int = 100
    priority: int = 1  # 1 (lowest) to 10 (highest)
    arrival_time: float = field(default_factory=time.time)
    timeout_s: float = 30.0
    
    # Tokenized data (filled by processor)
    input_ids: Optional[List[int]] = None
    attention_mask: Optional[List[int]] = None
    sequence_length: int = 0
    
    # Status tracking
    status: RequestStatus = RequestStatus.QUEUED
    batch_id: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    # Output
    output_tokens: Optional[List[int]] = None
    output_text: Optional[str] = None
    error: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering (higher priority first, then FIFO)."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.arrival_time < other.arrival_time
    
    @property
    def wait_time(self) -> float:
        """Time spent waiting in queue."""
        if self.start_time:
            return self.start_time - self.arrival_time
        return time.time() - self.arrival_time
    
    @property
    def is_expired(self) -> bool:
        """Check if request has timed out."""
        return (time.time() - self.arrival_time) > self.timeout_s


@dataclass
class Batch:
    """A batch of requests to process together."""
    batch_id: str
    requests: List[InferenceRequest]
    created_time: float = field(default_factory=time.time)
    
    # Batched tensors (filled during preparation)
    input_ids: Optional[Any] = None  # Tensor [batch_size, max_seq_len]
    attention_mask: Optional[Any] = None  # Tensor [batch_size, max_seq_len]
    position_ids: Optional[Any] = None  # Tensor [batch_size, max_seq_len]
    
    # Batch properties
    max_sequence_length: int = 0
    total_tokens: int = 0
    
    # Processing state
    current_position: int = 0  # For incremental generation
    is_complete: bool = False
    
    @property
    def size(self) -> int:
        return len(self.requests)
    
    @property
    def avg_priority(self) -> float:
        if not self.requests:
            return 0
        return sum(r.priority for r in self.requests) / len(self.requests)


@dataclass
class BatcherMetrics:
    """Metrics for batch processor."""
    total_requests: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_wait_time_ms: float = 0.0
    avg_batch_latency_ms: float = 0.0
    throughput_rps: float = 0.0  # Requests per second
    padding_overhead: float = 0.0  # Percentage of padded tokens


class BatchProcessor:
    """
    Batch processor for inference requests.
    
    Features:
    - Multiple batching strategies
    - Sequence length-aware batching
    - Priority queue for requests
    - Continuous batching support
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_sequence_length: int = 2048,
        max_batch_tokens: int = 16384,  # Total tokens in batch
        batch_timeout_ms: float = 100.0,
        strategy: BatchingStrategy = BatchingStrategy.ADAPTIVE,
        pad_token_id: int = 0,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize batch processor.
        
        Args:
            max_batch_size: Maximum requests per batch
            max_sequence_length: Maximum sequence length
            max_batch_tokens: Maximum total tokens in a batch
            batch_timeout_ms: Maximum wait time before forming batch
            strategy: Batching strategy to use
            pad_token_id: Token ID for padding
            tokenizer: Tokenizer instance (optional)
        """
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.max_batch_tokens = max_batch_tokens
        self.batch_timeout_ms = batch_timeout_ms
        self.strategy = strategy
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        
        # Request queue (priority queue)
        self._queue: PriorityQueue = PriorityQueue()
        self._pending_requests: Dict[str, InferenceRequest] = {}
        
        # Active batches
        self._active_batches: Dict[str, Batch] = {}
        self._batch_counter = 0
        
        # Metrics
        self.metrics = BatcherMetrics()
        self._completed_batches: List[Batch] = []
        
        # Synchronization
        self._lock = threading.RLock()
        self._batch_ready = threading.Event()
        
        # Background batching
        self._running = False
        self._batcher_thread: Optional[threading.Thread] = None
        self._batch_callback: Optional[Callable[[Batch], None]] = None
    
    def submit_request(self, request: InferenceRequest) -> str:
        """
        Submit a request for batched processing.
        
        Returns request_id for tracking.
        """
        with self._lock:
            # Tokenize if tokenizer available
            if self.tokenizer and request.input_ids is None:
                self._tokenize_request(request)
            
            # Add to queue
            self._queue.put(request)
            self._pending_requests[request.request_id] = request
            self.metrics.total_requests += 1
            
            # Signal batch formation
            self._batch_ready.set()
            
            return request.request_id
    
    def _tokenize_request(self, request: InferenceRequest) -> None:
        """Tokenize request prompt."""
        if self.tokenizer:
            encoded = self.tokenizer(
                request.prompt,
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt",
            )
            request.input_ids = encoded["input_ids"][0].tolist()
            request.attention_mask = encoded["attention_mask"][0].tolist()
            request.sequence_length = len(request.input_ids)
    
    def get_next_batch(self, timeout_s: float = 1.0) -> Optional[Batch]:
        """
        Get the next batch of requests to process.
        
        Blocks until a batch is ready or timeout.
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                batch = self._try_form_batch()
                if batch:
                    return batch
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout_s:
                # Force batch with available requests
                with self._lock:
                    return self._force_form_batch()
            
            # Wait for more requests
            remaining = timeout_s - elapsed
            self._batch_ready.wait(timeout=min(0.01, remaining))
            self._batch_ready.clear()
    
    def _try_form_batch(self) -> Optional[Batch]:
        """Try to form a batch based on current strategy."""
        if self._queue.empty():
            return None
        
        if self.strategy == BatchingStrategy.FIXED_SIZE:
            return self._form_fixed_size_batch()
        elif self.strategy == BatchingStrategy.FIXED_TIME:
            return self._form_fixed_time_batch()
        elif self.strategy == BatchingStrategy.ADAPTIVE:
            return self._form_adaptive_batch()
        elif self.strategy == BatchingStrategy.CONTINUOUS:
            return self._form_continuous_batch()
        
        return None
    
    def _form_fixed_size_batch(self) -> Optional[Batch]:
        """Form batch when we have max_batch_size requests."""
        if self._queue.qsize() < self.max_batch_size:
            return None
        
        requests = []
        for _ in range(self.max_batch_size):
            if self._queue.empty():
                break
            requests.append(self._queue.get())
        
        return self._create_batch(requests)
    
    def _form_fixed_time_batch(self) -> Optional[Batch]:
        """Form batch based on time window."""
        if self._queue.empty():
            return None
        
        # Check oldest request
        requests = list(self._pending_requests.values())
        if not requests:
            return None
        
        oldest = min(requests, key=lambda r: r.arrival_time)
        wait_ms = (time.time() - oldest.arrival_time) * 1000
        
        if wait_ms >= self.batch_timeout_ms:
            return self._force_form_batch()
        
        return None
    
    def _form_adaptive_batch(self) -> Optional[Batch]:
        """
        Adaptively form batch based on multiple factors:
        - Queue size
        - Wait time
        - Total tokens
        - Priority distribution
        """
        if self._queue.empty():
            return None
        
        pending = list(self._pending_requests.values())
        if not pending:
            return None
        
        # Check if we have high-priority requests waiting too long
        high_priority = [r for r in pending if r.priority >= 7]
        if high_priority:
            max_wait = max(r.wait_time for r in high_priority)
            if max_wait > self.batch_timeout_ms / 1000 / 2:
                # Form batch with high-priority requests
                return self._form_priority_batch(high_priority)
        
        # Check if we have enough requests
        if len(pending) >= self.max_batch_size:
            return self._force_form_batch()
        
        # Check wait time
        oldest = min(pending, key=lambda r: r.arrival_time)
        if oldest.wait_time * 1000 >= self.batch_timeout_ms:
            return self._force_form_batch()
        
        # Check total tokens
        total_tokens = sum(r.sequence_length for r in pending if r.sequence_length > 0)
        if total_tokens >= self.max_batch_tokens * 0.8:
            return self._force_form_batch()
        
        return None
    
    def _form_continuous_batch(self) -> Optional[Batch]:
        """
        Form batch for continuous batching.
        Can add new requests to running batch at iteration boundaries.
        """
        # For continuous batching, we form smaller batches more frequently
        if self._queue.qsize() >= max(1, self.max_batch_size // 2):
            return self._force_form_batch()
        
        # Or if any request waiting too long
        for req in self._pending_requests.values():
            if req.wait_time * 1000 >= self.batch_timeout_ms / 2:
                return self._force_form_batch()
        
        return None
    
    def _form_priority_batch(self, priority_requests: List[InferenceRequest]) -> Batch:
        """Form batch prioritizing specific requests."""
        requests = []
        
        # Add priority requests first
        for req in priority_requests[:self.max_batch_size]:
            if req.request_id in self._pending_requests:
                requests.append(req)
                del self._pending_requests[req.request_id]
        
        # Fill remaining with other requests
        while len(requests) < self.max_batch_size and not self._queue.empty():
            req = self._queue.get()
            if req.request_id in self._pending_requests:
                requests.append(req)
                del self._pending_requests[req.request_id]
        
        return self._create_batch(requests)
    
    def _force_form_batch(self) -> Optional[Batch]:
        """Force form a batch with available requests."""
        if self._queue.empty():
            return None
        
        requests = []
        total_tokens = 0
        
        while len(requests) < self.max_batch_size and not self._queue.empty():
            req = self._queue.get()
            
            # Check token budget
            if req.sequence_length > 0:
                if total_tokens + req.sequence_length > self.max_batch_tokens:
                    # Put back and stop
                    self._queue.put(req)
                    break
                total_tokens += req.sequence_length
            
            requests.append(req)
            if req.request_id in self._pending_requests:
                del self._pending_requests[req.request_id]
        
        if not requests:
            return None
        
        return self._create_batch(requests)
    
    def _create_batch(self, requests: List[InferenceRequest]) -> Batch:
        """Create a batch from requests."""
        self._batch_counter += 1
        batch_id = f"batch_{self._batch_counter}"
        
        batch = Batch(
            batch_id=batch_id,
            requests=requests,
        )
        
        # Update request statuses
        for req in requests:
            req.status = RequestStatus.BATCHED
            req.batch_id = batch_id
        
        # Prepare batched tensors
        self._prepare_batch_tensors(batch)
        
        self._active_batches[batch_id] = batch
        self.metrics.total_batches += 1
        
        return batch
    
    def _prepare_batch_tensors(self, batch: Batch) -> None:
        """Prepare padded tensors for the batch."""
        if not batch.requests:
            return
        
        # Find max sequence length in batch
        max_len = max(
            (r.sequence_length for r in batch.requests if r.sequence_length > 0),
            default=1
        )
        max_len = min(max_len, self.max_sequence_length)
        batch.max_sequence_length = max_len
        
        # Prepare lists for batching
        input_ids_list = []
        attention_mask_list = []
        
        total_tokens = 0
        padded_tokens = 0
        
        for req in batch.requests:
            if req.input_ids is None:
                # Use placeholder
                ids = [self.pad_token_id] * max_len
                mask = [0] * max_len
            else:
                # Pad or truncate
                ids = req.input_ids[:max_len]
                mask = req.attention_mask[:max_len] if req.attention_mask else [1] * len(ids)
                
                # Pad to max_len
                pad_len = max_len - len(ids)
                if pad_len > 0:
                    ids = ids + [self.pad_token_id] * pad_len
                    mask = mask + [0] * pad_len
                    padded_tokens += pad_len
                
                total_tokens += len(req.input_ids)
            
            input_ids_list.append(ids)
            attention_mask_list.append(mask)
        
        batch.input_ids = input_ids_list
        batch.attention_mask = attention_mask_list
        batch.total_tokens = total_tokens
        
        # Update padding overhead metric
        total_batch_tokens = len(batch.requests) * max_len
        if total_batch_tokens > 0:
            overhead = padded_tokens / total_batch_tokens
            n = self.metrics.total_batches
            self.metrics.padding_overhead = (
                (self.metrics.padding_overhead * (n - 1) + overhead) / n
            )
    
    def complete_batch(
        self,
        batch_id: str,
        outputs: Optional[List[Any]] = None,
        errors: Optional[List[str]] = None
    ) -> None:
        """Mark batch as complete and update request statuses."""
        with self._lock:
            if batch_id not in self._active_batches:
                return
            
            batch = self._active_batches[batch_id]
            batch.is_complete = True
            
            for i, req in enumerate(batch.requests):
                req.completion_time = time.time()
                
                if errors and i < len(errors) and errors[i]:
                    req.status = RequestStatus.FAILED
                    req.error = errors[i]
                else:
                    req.status = RequestStatus.COMPLETED
                    if outputs and i < len(outputs):
                        req.output_tokens = outputs[i]
            
            # Update metrics
            self._update_metrics(batch)
            
            del self._active_batches[batch_id]
            self._completed_batches.append(batch)
            
            # Keep bounded
            if len(self._completed_batches) > 100:
                self._completed_batches = self._completed_batches[-100:]
    
    def _update_metrics(self, batch: Batch) -> None:
        """Update metrics after batch completion."""
        n = self.metrics.total_batches
        
        # Update average batch size
        self.metrics.avg_batch_size = (
            (self.metrics.avg_batch_size * (n - 1) + batch.size) / n
        )
        
        # Update average wait time
        avg_wait = sum(r.wait_time for r in batch.requests) / batch.size if batch.size > 0 else 0
        self.metrics.avg_wait_time_ms = (
            (self.metrics.avg_wait_time_ms * (n - 1) + avg_wait * 1000) / n
        )
        
        # Update batch latency
        if batch.requests:
            batch_start = min(r.start_time or r.arrival_time for r in batch.requests)
            batch_end = max(r.completion_time or time.time() for r in batch.requests)
            latency_ms = (batch_end - batch_start) * 1000
            self.metrics.avg_batch_latency_ms = (
                (self.metrics.avg_batch_latency_ms * (n - 1) + latency_ms) / n
            )
    
    def get_request_status(self, request_id: str) -> Optional[RequestStatus]:
        """Get status of a specific request."""
        with self._lock:
            if request_id in self._pending_requests:
                return self._pending_requests[request_id].status
            
            for batch in self._active_batches.values():
                for req in batch.requests:
                    if req.request_id == request_id:
                        return req.status
            
            for batch in self._completed_batches:
                for req in batch.requests:
                    if req.request_id == request_id:
                        return req.status
        
        return None
    
    def get_metrics(self) -> BatcherMetrics:
        """Get current metrics."""
        with self._lock:
            # Calculate throughput
            if self._completed_batches:
                first_time = self._completed_batches[0].created_time
                last_time = time.time()
                duration = last_time - first_time
                if duration > 0:
                    total_requests = sum(b.size for b in self._completed_batches)
                    self.metrics.throughput_rps = total_requests / duration
            
            return BatcherMetrics(
                total_requests=self.metrics.total_requests,
                total_batches=self.metrics.total_batches,
                avg_batch_size=self.metrics.avg_batch_size,
                avg_wait_time_ms=self.metrics.avg_wait_time_ms,
                avg_batch_latency_ms=self.metrics.avg_batch_latency_ms,
                throughput_rps=self.metrics.throughput_rps,
                padding_overhead=self.metrics.padding_overhead,
            )
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self._lock:
            return {
                "pending_requests": len(self._pending_requests),
                "active_batches": len(self._active_batches),
                "queue_size": self._queue.qsize(),
                "strategy": self.strategy.value,
            }
    
    def start_background_batching(
        self,
        batch_callback: Callable[[Batch], None]
    ) -> None:
        """Start background batching thread."""
        self._batch_callback = batch_callback
        self._running = True
        self._batcher_thread = threading.Thread(
            target=self._batching_loop,
            daemon=True
        )
        self._batcher_thread.start()
    
    def stop_background_batching(self) -> None:
        """Stop background batching."""
        self._running = False
        self._batch_ready.set()
        if self._batcher_thread:
            self._batcher_thread.join(timeout=5.0)
    
    def _batching_loop(self) -> None:
        """Background batching loop."""
        while self._running:
            batch = self.get_next_batch(timeout_s=0.1)
            if batch and self._batch_callback:
                try:
                    self._batch_callback(batch)
                except Exception as e:
                    # Mark batch as failed
                    self.complete_batch(
                        batch.batch_id,
                        errors=[str(e)] * batch.size
                    )


class ContinuousBatcher:
    """
    Continuous batching implementation.
    Allows adding/removing requests at iteration boundaries.
    """
    
    def __init__(
        self,
        max_batch_size: int = 16,
        max_sequence_length: int = 2048,
    ):
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        
        # Current running batch
        self._running_requests: Dict[str, InferenceRequest] = {}
        self._waiting_requests: List[InferenceRequest] = []
        
        self._lock = threading.Lock()
    
    def add_request(self, request: InferenceRequest) -> bool:
        """Add request to waiting queue."""
        with self._lock:
            self._waiting_requests.append(request)
            return True
    
    def iteration_step(self) -> Tuple[List[InferenceRequest], List[InferenceRequest]]:
        """
        Called at each iteration boundary.
        Returns (active_requests, completed_requests).
        """
        with self._lock:
            completed = []
            
            # Check for completed requests
            for req_id, req in list(self._running_requests.items()):
                if req.status == RequestStatus.COMPLETED:
                    completed.append(req)
                    del self._running_requests[req_id]
            
            # Add new requests if space available
            while (
                len(self._running_requests) < self.max_batch_size
                and self._waiting_requests
            ):
                req = self._waiting_requests.pop(0)
                req.status = RequestStatus.PROCESSING
                req.start_time = time.time()
                self._running_requests[req.request_id] = req
            
            return list(self._running_requests.values()), completed
    
    def mark_request_complete(self, request_id: str) -> None:
        """Mark a request as complete."""
        with self._lock:
            if request_id in self._running_requests:
                self._running_requests[request_id].status = RequestStatus.COMPLETED
                self._running_requests[request_id].completion_time = time.time()


# Plugin registration
try:
    from model.plugin_architecture import PluginRegistry
    
    @PluginRegistry.register("batcher", "default")
    class BatchProcessorPlugin:
        """Batch Processor Plugin."""
        
        def __init__(self, **kwargs):
            self.processor = BatchProcessor(**kwargs)
        
        def submit(self, request: InferenceRequest) -> str:
            return self.processor.submit_request(request)
        
        def get_batch(self, timeout: float = 1.0) -> Optional[Batch]:
            return self.processor.get_next_batch(timeout)
        
        def complete(self, batch_id: str, outputs=None, errors=None):
            self.processor.complete_batch(batch_id, outputs, errors)

except ImportError:
    pass
