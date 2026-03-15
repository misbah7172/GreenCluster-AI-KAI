"""
Async event bus for power threshold events.

Publishers emit :class:`ThresholdEvent` objects; subscribers receive them
via registered callbacks.  Thread-safe, non-blocking.

The :class:`EventBus` uses a background dispatcher thread and a bounded
:class:`queue.Queue` to decouple publishers from subscribers.
"""

import collections
import enum
import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ThresholdLevel(enum.Enum):
    """Power threshold classification relative to TDP."""

    OPTIMAL = "optimal"       # < 70 % TDP
    WARNING = "warning"       # >= 70 % and < 80 % TDP
    CRITICAL = "critical"     # >= 80 % TDP


@dataclass
class ThresholdEvent:
    """A single power threshold event."""

    level: ThresholdLevel
    node_name: str
    power_w: float
    tdp_w: float
    tdp_pct: float
    timestamp: str = ""       # ISO 8601 — filled automatically if empty
    gpu_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "node_name": self.node_name,
            "power_w": self.power_w,
            "tdp_w": self.tdp_w,
            "tdp_pct": self.tdp_pct,
            "timestamp": self.timestamp,
            "gpu_index": self.gpu_index,
            "metadata": self.metadata,
        }


class EventBus:
    """Thread-safe pub/sub for :class:`ThresholdEvent` objects.

    Parameters
    ----------
    max_queue_size : int
        Upper bound on the internal dispatch queue (default 1000).
        Publish calls that would exceed this limit are silently dropped.
    history_size : int
        Number of recent events retained for :meth:`get_event_history`
        (default 200).
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        history_size: int = 200,
    ):
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._subscribers: List[
            tuple  # (callback, Optional[ThresholdLevel])
        ] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._history: collections.deque = collections.deque(maxlen=history_size)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background dispatcher thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="event-bus",
        )
        self._thread.start()
        logger.info("EventBus started")

    def stop(self) -> None:
        """Stop the dispatcher and drain remaining events."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("EventBus stopped")

    # ------------------------------------------------------------------
    # Pub / Sub
    # ------------------------------------------------------------------

    def publish(self, event: ThresholdEvent) -> None:
        """Non-blocking publish.  Drops the event if the queue is full."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("EventBus queue full — dropping event %s", event.level.value)

    def subscribe(
        self,
        callback: Callable[[ThresholdEvent], None],
        level_filter: Optional[ThresholdLevel] = None,
    ) -> None:
        """Register a subscriber callback.

        Parameters
        ----------
        callback : callable
            Invoked with a :class:`ThresholdEvent` on each matching event.
        level_filter : ThresholdLevel, optional
            If provided, the callback is only invoked for events of this
            level.  ``None`` means all events.
        """
        with self._lock:
            self._subscribers.append((callback, level_filter))

    def unsubscribe(self, callback: Callable) -> None:
        """Remove all registrations for *callback*."""
        with self._lock:
            self._subscribers = [
                (cb, lf) for (cb, lf) in self._subscribers if cb is not callback
            ]

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_event_history(self, n: int = 100) -> List[ThresholdEvent]:
        """Return the last *n* events (or all if *n* ≥ history size)."""
        with self._lock:
            items = list(self._history)
        if n <= 0 or n >= len(items):
            return items
        return items[-n:]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        """Background: dequeue events and deliver to matching subscribers."""
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Record in history
            with self._lock:
                self._history.append(event)
                subscribers = list(self._subscribers)

            for callback, level_filter in subscribers:
                if level_filter is not None and event.level != level_filter:
                    continue
                try:
                    callback(event)
                except Exception:
                    logger.exception("Subscriber %s raised an error", callback)
