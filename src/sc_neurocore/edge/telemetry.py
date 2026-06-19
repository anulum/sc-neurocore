# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Runtime Telemetry (ported from tinysc_riscv/telemetry.rs)

"""On-device spike-rate and utilization counters for runtime monitoring.

Zero-allocation ring buffers for tracking spike rates, activity metrics,
and health counters. Mirrors the bare-metal Rust implementation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


class TelemetryRing:
    """Fixed-size ring buffer for telemetry samples (u32 values)."""

    def __init__(self, capacity: int = 256):
        self._cap = max(capacity, 1)
        self._buf = [0] * self._cap
        self._write_idx = 0
        self._count = 0
        self._lock = threading.Lock()

    def push(self, value: int) -> None:
        with self._lock:
            self._buf[self._write_idx % self._cap] = value
            self._write_idx += 1
            if self._count < self._cap:
                self._count += 1

    def mean(self) -> float:
        with self._lock:
            if self._count == 0:
                return 0.0
            n = self._count
            start = (self._write_idx - n) % self._cap
            total = 0
            for i in range(n):
                total += self._buf[(start + i) % self._cap]
            return total / n

    def last(self) -> int:
        with self._lock:
            if self._count == 0:
                return 0
            return self._buf[(self._write_idx - 1) % self._cap]

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def capacity(self) -> int:
        return self._cap


@dataclass
class LayerTelemetry:
    """Telemetry counters for a single SC layer."""

    layer_id: str = ""
    spike_count: int = 0
    tick_count: int = 0
    total_popcount: int = 0
    spike_rate_ring: TelemetryRing = field(default_factory=lambda: TelemetryRing(64))
    utilization_ring: TelemetryRing = field(default_factory=lambda: TelemetryRing(64))

    def record_tick(self, n_spikes: int, n_neurons: int) -> None:
        """Record one tick's worth of activity."""
        self.tick_count += 1
        self.spike_count += n_spikes
        self.spike_rate_ring.push(n_spikes)
        if n_neurons > 0:
            utilization = (n_spikes * 100) // n_neurons
            self.utilization_ring.push(utilization)

    @property
    def mean_spike_rate(self) -> float:
        return self.spike_rate_ring.mean()

    @property
    def mean_utilization(self) -> float:
        return self.utilization_ring.mean()

    @property
    def lifetime_spike_rate(self) -> float:
        if self.tick_count == 0:
            return 0.0
        return self.spike_count / self.tick_count


@dataclass
class DeviceTelemetry:
    """Aggregate telemetry for the full device/network."""

    layers: dict[str, LayerTelemetry] = field(default_factory=dict)
    total_ticks: int = 0
    total_spikes: int = 0
    error_count: int = 0

    def get_layer(self, layer_id: str) -> LayerTelemetry:
        if layer_id not in self.layers:
            self.layers[layer_id] = LayerTelemetry(layer_id=layer_id)
        return self.layers[layer_id]

    def record(self, layer_id: str, n_spikes: int, n_neurons: int) -> None:
        self.total_ticks += 1
        self.total_spikes += n_spikes
        self.get_layer(layer_id).record_tick(n_spikes, n_neurons)

    def summary(self) -> dict[str, Any]:
        return {
            "total_ticks": self.total_ticks,
            "total_spikes": self.total_spikes,
            "error_count": self.error_count,
            "layers": {
                lid: {
                    "spike_count": lt.spike_count,
                    "tick_count": lt.tick_count,
                    "mean_spike_rate": lt.mean_spike_rate,
                    "mean_utilization": lt.mean_utilization,
                }
                for lid, lt in self.layers.items()
            },
        }
