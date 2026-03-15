"""
Power Threshold Service.

Evaluates GPU power samples against TDP-relative thresholds and
publishes :class:`~monitoring.event_bus.ThresholdEvent` objects to an
:class:`~monitoring.event_bus.EventBus`.

Thresholds
----------
- **OPTIMAL**  : power < 70 % TDP
- **WARNING**  : 70 % TDP <= power < 80 % TDP
- **CRITICAL** : power >= 80 % TDP
"""

import logging
import threading
from typing import Optional

from monitoring.event_bus import EventBus, ThresholdEvent, ThresholdLevel
from monitoring.gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)

OPTIMAL_THRESHOLD_PCT = 70.0
WARNING_THRESHOLD_PCT = 80.0


class PowerThresholdService:
    """Monitors GPU power and publishes threshold events.

    A background thread periodically reads the latest sample from the
    :class:`GPUMonitor` ring buffer.  When the classification changes
    (or when the first sample is evaluated), a :class:`ThresholdEvent`
    is published to the :class:`EventBus`.

    Parameters
    ----------
    gpu_monitor : GPUMonitor
        The GPU monitor to read from.
    event_bus : EventBus
        Event bus to publish threshold events to.
    node_name : str
        Name of this node (for event metadata).
    tdp_watts : float
        TDP in watts.  If 0, uses ``gpu_monitor.get_tdp_watts()``.
    check_interval : float
        How often to evaluate thresholds (seconds).  Default 0.5.
    """

    def __init__(
        self,
        gpu_monitor: GPUMonitor,
        event_bus: EventBus,
        node_name: str = "local",
        tdp_watts: float = 0.0,
        check_interval: float = 0.5,
    ):
        self._gpu_monitor = gpu_monitor
        self._event_bus = event_bus
        self._node_name = node_name
        self._check_interval = check_interval

        self._tdp_watts = tdp_watts if tdp_watts > 0 else gpu_monitor.get_tdp_watts()
        self._current_level: Optional[ThresholdLevel] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start threshold evaluation in a background thread."""
        if self._tdp_watts <= 0:
            # Try once more at start time (monitor may now have detected TDP)
            self._tdp_watts = self._gpu_monitor.get_tdp_watts()
        if self._tdp_watts <= 0:
            logger.warning(
                "TDP unknown — PowerThresholdService will not classify events"
            )
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._evaluate_loop, daemon=True, name="threshold-svc",
        )
        self._thread.start()
        logger.info(
            "PowerThresholdService started (TDP=%.1fW, interval=%.2fs)",
            self._tdp_watts, self._check_interval,
        )

    def stop(self) -> None:
        """Stop the threshold evaluation thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._check_interval * 3)
            self._thread = None
        logger.info("PowerThresholdService stopped")

    def get_current_level(self) -> Optional[ThresholdLevel]:
        """Return the most recent threshold level (or ``None``)."""
        return self._current_level

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, power_w: float) -> ThresholdLevel:
        """Classify *power_w* relative to TDP.

        Returns :attr:`ThresholdLevel.CRITICAL` when the TDP is unknown
        or zero (fail-safe).
        """
        if self._tdp_watts <= 0:
            return ThresholdLevel.CRITICAL

        pct = power_w / self._tdp_watts * 100.0
        if pct >= WARNING_THRESHOLD_PCT:
            return ThresholdLevel.CRITICAL
        if pct >= OPTIMAL_THRESHOLD_PCT:
            return ThresholdLevel.WARNING
        return ThresholdLevel.OPTIMAL

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_loop(self) -> None:
        """Periodically check the latest GPU sample against thresholds."""
        while not self._stop_event.is_set():
            try:
                recent = self._gpu_monitor.get_recent_samples(1)
                if recent:
                    sample = recent[-1]
                    power_w = sample.get("power_w", 0.0)
                    new_level = self.classify(power_w)

                    # Publish on state change (or first evaluation)
                    if new_level != self._current_level:
                        self._current_level = new_level
                        tdp_pct = (
                            round(power_w / self._tdp_watts * 100.0, 2)
                            if self._tdp_watts > 0
                            else 0.0
                        )
                        event = ThresholdEvent(
                            level=new_level,
                            node_name=self._node_name,
                            power_w=power_w,
                            tdp_w=self._tdp_watts,
                            tdp_pct=tdp_pct,
                            gpu_index=self._gpu_monitor.gpu_index,
                        )
                        self._event_bus.publish(event)
                        logger.info(
                            "Threshold state → %s (%.1fW, %.1f%% TDP)",
                            new_level.value, power_w, tdp_pct,
                        )
            except Exception:
                logger.exception("Error evaluating power threshold")

            self._stop_event.wait(self._check_interval)
