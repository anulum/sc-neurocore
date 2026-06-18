# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AER Interconnect Router (ported from interconnect/main.go)

"""AER-over-UDP multi-FPGA spike router.

Provides dynamic routing, ACK-based reliability, sequence tracking, and
per-route statistics.

Protocol: each SpikePacket is a 28-byte big-endian binary frame; ACKs are
8-byte sequence echoes.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass


PACKET_FORMAT = (
    ">IIQIq"  # source_id(u32) target_id(u32) timestamp(u64) spike_len(u32) sequence(i64)
)
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)  # 28 bytes
ACK_FORMAT = ">q"
ACK_SIZE = 8


@dataclass
class SpikePacket:
    """Wire format for an AER spike event (28 bytes big-endian)."""

    source_id: int = 0
    target_id: int = 0
    timestamp: int = 0
    spike_len: int = 0
    sequence: int = 0

    def encode(self) -> bytes:
        """Serialize to 28-byte big-endian frame."""
        return struct.pack(
            PACKET_FORMAT,
            self.source_id,
            self.target_id,
            self.timestamp,
            self.spike_len,
            self.sequence,
        )

    @classmethod
    def decode(cls, data: bytes) -> SpikePacket:
        """Deserialize from 28-byte big-endian frame."""
        src, tgt, ts, slen, seq = struct.unpack(PACKET_FORMAT, data[:PACKET_SIZE])
        return cls(source_id=src, target_id=tgt, timestamp=ts, spike_len=slen, sequence=seq)


@dataclass
class RouteStats:
    """Per-route delivery statistics."""

    dispatched: int = 0
    acked: int = 0
    dropped: int = 0


class AERRouter:
    """Manages route registration, spike dispatch, and ACK tracking.

    This is a pure-Python simulation/client. For high-performance
    UDP routing, use the Go server (hil_debugger/interconnect).
    """

    def __init__(self) -> None:
        self._routes: dict[int, str] = {}
        self._stats: dict[int, RouteStats] = {}
        self._pending: dict[int, float] = {}
        self._total_sent = 0
        self._total_acked = 0
        self._lock = threading.Lock()

    def register_route(self, neuron_id: int, addr: str) -> None:
        """Map a neuron ID to a destination address (host:port)."""
        with self._lock:
            self._routes[neuron_id] = addr
            if neuron_id not in self._stats:
                self._stats[neuron_id] = RouteStats()

    def unregister_route(self, neuron_id: int) -> None:
        """Remove a route for the given neuron ID."""
        with self._lock:
            self._routes.pop(neuron_id, None)

    @property
    def route_count(self) -> int:
        """Return the number of currently registered neuron routes."""
        with self._lock:
            return len(self._routes)

    def dispatch_spike(self, packet: SpikePacket) -> bool:
        """Dispatch a spike packet to the registered target.

        Returns True if the route exists and dispatch succeeded.
        Does not perform actual UDP send in simulation mode.
        """
        with self._lock:
            target = self._routes.get(packet.target_id)
            stats = self._stats.get(packet.target_id)
            if target is None:
                return False
            self._pending[packet.sequence] = time.monotonic()
            if stats:
                stats.dispatched += 1
            self._total_sent += 1
            return True

    def ack_received(self, seq: int) -> None:
        """Process an ACK for the given sequence number."""
        with self._lock:
            self._pending.pop(seq, None)
            self._total_acked += 1

    @property
    def pending_count(self) -> int:
        """Return the number of dispatched packets awaiting ACKs."""
        with self._lock:
            return len(self._pending)

    @property
    def total_sent(self) -> int:
        """Return the total number of packets accepted for dispatch."""
        with self._lock:
            return self._total_sent

    @property
    def total_acked(self) -> int:
        """Return the total number of ACKs processed by the router."""
        with self._lock:
            return self._total_acked

    def get_stats(self, neuron_id: int) -> RouteStats | None:
        """Return a defensive copy of per-route statistics when present."""
        with self._lock:
            s = self._stats.get(neuron_id)
            return RouteStats(s.dispatched, s.acked, s.dropped) if s else None
