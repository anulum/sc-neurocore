# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HIL Debugger Client (ported from hil_debugger/main.go)

"""Hardware-in-the-Loop debugger telemetry components.

Provides ring buffer, layer aggregation, error budget, correlation
window, precision tracking, event filtering, trigger conditions,
rate limiting, health checks, and CSV/JSON export.

NOTE: This is a pure-Python simulation/testing reference.
For high-performance, real-time edge telemetry, use the compiled Go
server daemon (`sc_neurocore.debug.hil_server.HILServerDaemon`).
"""

from __future__ import annotations

import csv
import io
import json
import threading
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SpikeEvent:
    """Single telemetry sample from an FPGA or simulator.

    Attributes
    ----------
    timestamp : int
        Sample time, in the source's unit.
    layer_id : str
        Layer of the neuron.
    neuron_id : int
        Neuron index.
    correlation : float
        Stochastic-computing correlation reported with the sample.
    popcount : int
        Ones counted in the sample's bitstream.
    precision : float
        Precision reported with the sample.
    sequence : int
        Sequence number of the sample.
    """

    timestamp: int = 0
    layer_id: str = ""
    neuron_id: int = 0
    correlation: float = 0.0
    popcount: int = 0
    precision: float = 0.0
    sequence: int = 0


class SpikeRingBuffer:
    """Fixed-capacity circular buffer for SpikeEvent telemetry.

    Lock-protected overwrite-on-full. Mirrors Go RingBuffer.

    Parameters
    ----------
    capacity : int
        Number of events kept; values below 1 become 1.
    """

    def __init__(self, capacity: int = 1024):
        self._cap = max(capacity, 1)
        self._data: list[SpikeEvent] = [SpikeEvent() for _ in range(self._cap)]
        self._head = 0
        self._lock = threading.Lock()

    def push(self, evt: SpikeEvent) -> None:
        """Store one event, overwriting the oldest when the buffer is full."""
        with self._lock:
            self._data[self._head % self._cap] = evt
            self._head += 1

    def snapshot(self, n: int = 0) -> list[SpikeEvent]:
        """Return the most recent events, oldest first.

        Parameters
        ----------
        n : int
            How many to return; 0, a negative number or one at least the number
            held returns all that are held.

        Returns
        -------
        list of SpikeEvent
            Up to ``capacity`` events; empty before the first push.
        """
        with self._lock:
            if self._head == 0:
                return []
            count = min(self._head, self._cap)
            if 0 < n < count:
                count = n
            result = []
            for i in range(count):
                idx = (self._head - count + i) % self._cap
                result.append(self._data[idx])
            return result

    @property
    def head(self) -> int:
        """Number of events pushed so far, never reduced by overwriting."""
        return self._head

    @property
    def capacity(self) -> int:
        """Number of events the buffer keeps."""
        return self._cap


class LayerAggregator:
    """Per-layer running statistics collector.

    Each layer's statistics are a dictionary with the event count, the sums of
    correlation, precision and popcount, the smallest precision and the largest
    correlation seen.
    """

    def __init__(self) -> None:
        self._layers: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def record(self, evt: SpikeEvent) -> None:
        """Add one event to its layer's statistics, starting them on the layer's first event."""
        with self._lock:
            ls = self._layers.get(evt.layer_id)
            if ls is None:
                ls = {
                    "layer_id": evt.layer_id,
                    "event_count": 0,
                    "sum_correlation": 0.0,
                    "sum_precision": 0.0,
                    "sum_popcount": 0,
                    "min_precision": evt.precision,
                    "max_correlation": evt.correlation,
                }
                self._layers[evt.layer_id] = ls
            ls["event_count"] += 1
            ls["sum_correlation"] += evt.correlation
            ls["sum_precision"] += evt.precision
            ls["sum_popcount"] += evt.popcount
            if evt.precision < ls["min_precision"]:
                ls["min_precision"] = evt.precision
            if evt.correlation > ls["max_correlation"]:
                ls["max_correlation"] = evt.correlation

    def get(self, layer_id: str) -> Optional[dict[str, Any]]:
        """Return a copy of one layer's statistics, or ``None`` for a layer not seen."""
        with self._lock:
            ls = self._layers.get(layer_id)
            return dict(ls) if ls else None

    def all(self) -> dict[str, dict[str, Any]]:
        """Return a copy of every layer's statistics, keyed by layer."""
        with self._lock:
            return {k: dict(v) for k, v in self._layers.items()}

    @staticmethod
    def mean_correlation(ls: dict[str, Any]) -> float:
        """Return a layer's mean correlation; 0.0 for statistics without events."""
        if ls["event_count"] == 0:
            return 0.0
        return float(ls["sum_correlation"] / ls["event_count"])

    @staticmethod
    def mean_precision(ls: dict[str, Any]) -> float:
        """Return a layer's mean precision; 0.0 for statistics without events."""
        if ls["event_count"] == 0:
            return 0.0
        return float(ls["sum_precision"] / ls["event_count"])


@dataclass
class ErrorBudget:
    """Threshold-based alerting for precision/correlation bounds.

    Attributes
    ----------
    min_precision : float
        Precision below which an event violates the budget.
    max_correlation : float
        Correlation above which an event violates the budget.
    violations : int
        Violating events counted so far.
    """

    min_precision: float = 0.90
    max_correlation: float = 0.20
    violations: int = 0

    def check(self, evt: SpikeEvent) -> bool:
        """Return whether an event violates the budget, counting it if so.

        An event violates the budget when its precision is below the minimum or its
        correlation above the maximum; it is counted once either way.
        """
        violated = False
        if evt.precision < self.min_precision:
            violated = True
        if evt.correlation > self.max_correlation:
            violated = True
        if violated:
            self.violations += 1
        return violated


class CorrelationWindow:
    """Sliding window for correlation values.

    Parameters
    ----------
    size : int
        Number of values kept; values below 1 become 1.
    """

    def __init__(self, size: int = 128):
        self._cap = max(size, 1)
        self._values = [0.0] * self._cap
        self._pos = 0
        self._full = False

    def add(self, v: float) -> None:
        """Store one value, overwriting the oldest when the window is full."""
        self._values[self._pos] = v
        self._pos = (self._pos + 1) % self._cap
        if self._pos == 0:
            self._full = True

    @property
    def count(self) -> int:
        """Number of values held, at most the window size."""
        return self._cap if self._full else self._pos

    def mean(self) -> float:
        """Return the mean of the held values; 0.0 when empty."""
        n = self.count
        if n == 0:
            return 0.0
        return sum(self._values[:n]) / n

    def max(self) -> float:
        """Return the largest held value; 0.0 when empty."""
        n = self.count
        if n == 0:
            return 0.0
        return max(self._values[:n])


@dataclass
class PrecisionTracker:
    """Exponential moving average of precision.

    Attributes
    ----------
    alpha : float
        Weight of the newest value.
    ema : float
        The average; the first value itself after the first update.
    count : int
        Updates so far.
    """

    alpha: float = 0.05
    ema: float = 0.0
    count: int = 0

    def update(self, precision: float) -> None:
        """Fold one precision value into the average."""
        self.count += 1
        if self.count == 1:
            self.ema = precision
            return
        self.ema = self.alpha * precision + (1 - self.alpha) * self.ema


@dataclass
class EventFilter:
    """Selects events matching criteria.

    Attributes
    ----------
    layer_id : str
        Layer an event must belong to; empty matches every layer.
    min_neuron, max_neuron : int
        Inclusive neuron range, used only when ``has_neuron`` is set.
    has_neuron : bool
        Whether the neuron range applies.
    """

    layer_id: str = ""
    min_neuron: int = 0
    max_neuron: int = 0
    has_neuron: bool = False

    def match(self, evt: SpikeEvent) -> bool:
        """Return whether an event passes the layer and, if set, the neuron range."""
        if self.layer_id and evt.layer_id != self.layer_id:
            return False
        if self.has_neuron:
            if evt.neuron_id < self.min_neuron or evt.neuron_id > self.max_neuron:
                return False
        return True


def filter_events(events: list[SpikeEvent], f: EventFilter) -> list[SpikeEvent]:
    """Apply a filter to a list of events."""
    return [e for e in events if f.match(e)]


@dataclass
class TriggerCondition:
    """Conditional breakpoint for debugger.

    Attributes
    ----------
    min_correlation : float
        A correlation at or above it fires the trigger; 0 disables this test.
    max_precision : float
        A precision at or below it fires the trigger; 0 disables this test.
    layer_id : str
        Layer the trigger watches; empty watches every layer.
    armed : bool
        Whether the trigger fires at all.

    Notes
    -----
    With both thresholds at their default 0 the trigger never fires.
    """

    min_correlation: float = 0.0
    max_precision: float = 0.0
    layer_id: str = ""
    armed: bool = True

    def evaluate(self, evt: SpikeEvent) -> bool:
        """Return whether an event fires the trigger."""
        if not self.armed:
            return False
        if self.layer_id and evt.layer_id != self.layer_id:
            return False
        if self.min_correlation > 0 and evt.correlation >= self.min_correlation:
            return True
        return bool(self.max_precision > 0 and evt.precision <= self.max_precision)


class TriggerLog:
    """Records fired trigger events for post-mortem analysis.

    Attributes
    ----------
    entries : list of SpikeEvent
        The fired events, in order, never trimmed.
    """

    def __init__(self) -> None:
        self.entries: list[SpikeEvent] = []
        self._lock = threading.Lock()

    def fire(self, evt: SpikeEvent) -> None:
        """Append one fired event."""
        with self._lock:
            self.entries.append(evt)

    @property
    def count(self) -> int:
        """Number of fired events recorded."""
        with self._lock:
            return len(self.entries)


class RateLimiter:
    """Token-bucket rate limiter for high-speed streams.

    The bucket starts full and is refilled only by :meth:`refill`; nothing refills
    it over time.

    Parameters
    ----------
    capacity : int
        Largest number of tokens, and the number at the start.
    """

    def __init__(self, capacity: int):
        self._tokens = capacity
        self._capacity = capacity
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Take one token if there is one, and return whether one was taken."""
        with self._lock:
            if self._tokens > 0:
                self._tokens -= 1
                return True
            return False

    def refill(self, n: int) -> None:
        """Add ``n`` tokens, up to the capacity."""
        with self._lock:
            self._tokens = min(self._tokens + n, self._capacity)

    @property
    def available(self) -> int:
        """Number of tokens left."""
        with self._lock:
            return self._tokens


@dataclass
class HealthStatus:
    """Debugger health snapshot.

    Attributes
    ----------
    status : str
        ``"healthy"`` or ``"buffer_pressure"``.
    events_per_sec : float
        Events received per second of uptime.
    buffer_usage : float
        Fraction of the ring buffer filled, from 0 to 1.
    clients_active : int
        Connected clients, as given.
    """

    status: str = "healthy"
    events_per_sec: float = 0.0
    buffer_usage: float = 0.0
    clients_active: int = 0


def check_health(
    events_received: int,
    uptime_seconds: int,
    buffer_head: int,
    buffer_capacity: int,
    clients_active: int = 0,
) -> HealthStatus:
    """Compute health status from telemetry metrics.

    As the Go daemon computes it: the buffer usage is min(head, capacity) /
    capacity, so once the ring buffer has filled once it stays at 1.0 and the
    status stays ``"buffer_pressure"`` (usage above 0.95) while the buffer is in
    use.

    Parameters
    ----------
    events_received : int
        Events received since the start.
    uptime_seconds : int
        Seconds since the start; 0 gives 0 events per second.
    buffer_head : int
        The ring buffer's head counter.
    buffer_capacity : int
        The ring buffer's capacity; 0 gives a usage of 0.
    clients_active : int
        Connected clients.

    Returns
    -------
    HealthStatus
        The snapshot.
    """
    usage = 0.0
    if buffer_capacity > 0:
        used = min(buffer_head, buffer_capacity)
        usage = used / buffer_capacity
    eps = events_received / uptime_seconds if uptime_seconds > 0 else 0.0
    status = "buffer_pressure" if usage > 0.95 else "healthy"
    return HealthStatus(
        status=status,
        events_per_sec=eps,
        buffer_usage=usage,
        clients_active=clients_active,
    )


def export_csv(events: list[SpikeEvent]) -> str:
    """Export events to CSV string.

    One header row (``timestamp``, ``layer_id``, ``neuron_id``, ``correlation``,
    ``popcount``, ``precision``, ``sequence``), then one row per event with
    correlation and precision at six decimals.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["timestamp", "layer_id", "neuron_id", "correlation", "popcount", "precision", "sequence"]
    )
    for e in events:
        writer.writerow(
            [
                e.timestamp,
                e.layer_id,
                e.neuron_id,
                f"{e.correlation:.6f}",
                e.popcount,
                f"{e.precision:.6f}",
                e.sequence,
            ]
        )
    return buf.getvalue()


def export_json(events: list[SpikeEvent]) -> str:
    """Export events to JSON array string.

    The keys follow the Go daemon's JSON (``ts`` and ``seq`` where the CSV says
    ``timestamp`` and ``sequence``); indented by two spaces.
    """
    data = [
        {
            "ts": e.timestamp,
            "layer_id": e.layer_id,
            "neuron_id": e.neuron_id,
            "correlation": e.correlation,
            "popcount": e.popcount,
            "precision": e.precision,
            "seq": e.sequence,
        }
        for e in events
    ]
    return json.dumps(data, indent=2)
