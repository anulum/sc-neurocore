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
    """Single telemetry sample from an FPGA or simulator."""

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
    """

    def __init__(self, capacity: int = 1024):
        self._cap = max(capacity, 1)
        self._data: list[SpikeEvent] = [SpikeEvent() for _ in range(self._cap)]
        self._head = 0
        self._lock = threading.Lock()

    def push(self, evt: SpikeEvent) -> None:
        with self._lock:
            self._data[self._head % self._cap] = evt
            self._head += 1

    def snapshot(self, n: int = 0) -> list[SpikeEvent]:
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
        return self._head

    @property
    def capacity(self) -> int:
        return self._cap


class LayerAggregator:
    """Per-layer running statistics collector."""

    def __init__(self) -> None:
        self._layers: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def record(self, evt: SpikeEvent) -> None:
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
        with self._lock:
            ls = self._layers.get(layer_id)
            return dict(ls) if ls else None

    def all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._layers.items()}

    @staticmethod
    def mean_correlation(ls: dict[str, Any]) -> float:
        if ls["event_count"] == 0:
            return 0.0
        return float(ls["sum_correlation"] / ls["event_count"])

    @staticmethod
    def mean_precision(ls: dict[str, Any]) -> float:
        if ls["event_count"] == 0:
            return 0.0
        return float(ls["sum_precision"] / ls["event_count"])


@dataclass
class ErrorBudget:
    """Threshold-based alerting for precision/correlation bounds."""

    min_precision: float = 0.90
    max_correlation: float = 0.20
    violations: int = 0

    def check(self, evt: SpikeEvent) -> bool:
        violated = False
        if evt.precision < self.min_precision:
            violated = True
        if evt.correlation > self.max_correlation:
            violated = True
        if violated:
            self.violations += 1
        return violated


class CorrelationWindow:
    """Sliding window for correlation values."""

    def __init__(self, size: int = 128):
        self._cap = max(size, 1)
        self._values = [0.0] * self._cap
        self._pos = 0
        self._full = False

    def add(self, v: float) -> None:
        self._values[self._pos] = v
        self._pos = (self._pos + 1) % self._cap
        if self._pos == 0:
            self._full = True

    @property
    def count(self) -> int:
        return self._cap if self._full else self._pos

    def mean(self) -> float:
        n = self.count
        if n == 0:
            return 0.0
        return sum(self._values[:n]) / n

    def max(self) -> float:
        n = self.count
        if n == 0:
            return 0.0
        return max(self._values[:n])


@dataclass
class PrecisionTracker:
    """Exponential moving average of precision."""

    alpha: float = 0.05
    ema: float = 0.0
    count: int = 0

    def update(self, precision: float) -> None:
        self.count += 1
        if self.count == 1:
            self.ema = precision
            return
        self.ema = self.alpha * precision + (1 - self.alpha) * self.ema


@dataclass
class EventFilter:
    """Selects events matching criteria."""

    layer_id: str = ""
    min_neuron: int = 0
    max_neuron: int = 0
    has_neuron: bool = False

    def match(self, evt: SpikeEvent) -> bool:
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
    """Conditional breakpoint for debugger."""

    min_correlation: float = 0.0
    max_precision: float = 0.0
    layer_id: str = ""
    armed: bool = True

    def evaluate(self, evt: SpikeEvent) -> bool:
        if not self.armed:
            return False
        if self.layer_id and evt.layer_id != self.layer_id:
            return False
        if self.min_correlation > 0 and evt.correlation >= self.min_correlation:
            return True
        return bool(self.max_precision > 0 and evt.precision <= self.max_precision)


class TriggerLog:
    """Records fired trigger events for post-mortem analysis."""

    def __init__(self) -> None:
        self.entries: list[SpikeEvent] = []
        self._lock = threading.Lock()

    def fire(self, evt: SpikeEvent) -> None:
        with self._lock:
            self.entries.append(evt)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.entries)


class RateLimiter:
    """Token-bucket rate limiter for high-speed streams."""

    def __init__(self, capacity: int):
        self._tokens = capacity
        self._capacity = capacity
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._tokens > 0:
                self._tokens -= 1
                return True
            return False

    def refill(self, n: int) -> None:
        with self._lock:
            self._tokens = min(self._tokens + n, self._capacity)

    @property
    def available(self) -> int:
        with self._lock:
            return self._tokens


@dataclass
class HealthStatus:
    """Debugger health snapshot."""

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
    """Compute health status from telemetry metrics."""
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
    """Export events to CSV string."""
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
    """Export events to JSON array string."""
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
