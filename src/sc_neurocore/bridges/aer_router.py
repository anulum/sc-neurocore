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

import secrets
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace

PACKET_FORMAT = (
    ">IIQIq"  # source_id(u32) target_id(u32) timestamp(u64) spike_len(u32) sequence(i64)
)
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)  # 28 bytes
ACK_FORMAT = ">q"
ACK_SIZE = 8
_MAX_PENDING = 4096
_ACK_TIMEOUT_SECONDS = 2.0
_MAX_DELIVERY_SEQUENCE = (1 << 63) - 1


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


@dataclass(frozen=True, slots=True)
class _PendingDispatch:
    """Destination and expiry captured when a simulated dispatch is accepted."""

    deadline: float
    upstream_sequence: int
    target_id: int
    address: str


class AERRouter:
    """Manages route registration, spike dispatch, and ACK tracking.

    This class models routing in memory. Use ``AERRoutingDaemon`` to
    supervise the Go UDP service.
    """

    def __init__(self) -> None:
        self._routes: dict[int, str] = {}
        self._stats: dict[int, RouteStats] = {}
        self._pending: OrderedDict[int, _PendingDispatch] = OrderedDict()
        self._inflight: dict[int, int] = {}
        self._delivery_base = secrets.randbelow(_MAX_DELIVERY_SEQUENCE)
        self._next_delivery = 0
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

        Returns True when the route exists and the sequence can be tracked.
        Use this method for immediate in-process dispatch and acknowledgement.
        """
        return self._track_dispatch(packet) is not None

    def dispatch_for_delivery(self, packet: SpikePacket) -> SpikePacket | None:
        """Prepare an outgoing packet with a distinct delivery sequence.

        Returns None when the route or pending capacity refuses the send.
        Echo the returned sequence with the downstream address to acknowledge it.
        """
        delivery = self._track_dispatch(packet)
        if delivery is None:
            return None
        return replace(packet, sequence=delivery)

    def _track_dispatch(self, packet: SpikePacket) -> int | None:
        """Track an accepted dispatch and return its non-reused delivery sequence."""
        with self._lock:
            now = time.monotonic()
            self._expire_pending(now)
            target = self._routes.get(packet.target_id)
            if target is None:
                return None
            stats = self._stats[packet.target_id]
            if (
                packet.sequence in self._inflight
                or len(self._pending) >= _MAX_PENDING
                or self._next_delivery == _MAX_DELIVERY_SEQUENCE
            ):
                stats.dropped += 1
                return None
            delivery = (self._delivery_base + self._next_delivery) % _MAX_DELIVERY_SEQUENCE + 1
            self._next_delivery += 1
            self._pending[delivery] = _PendingDispatch(
                now + _ACK_TIMEOUT_SECONDS, packet.sequence, packet.target_id, target
            )
            self._inflight[packet.sequence] = delivery
            stats.dispatched += 1
            self._total_sent += 1
            return delivery

    def _expire_pending(self, now: float) -> None:
        """Remove expired dispatches and account for drops while holding the lock."""
        while self._pending:
            delivery, dispatch = next(iter(self._pending.items()))
            if dispatch.deadline > now:
                break
            del self._pending[delivery]
            del self._inflight[dispatch.upstream_sequence]
            self._stats[dispatch.target_id].dropped += 1

    def ack_received(self, seq: int, *, address: str | None = None) -> None:
        """Acknowledge a delivery sequence and its downstream address.

        An immediate in-process acknowledgement accepts the upstream sequence
        with the address omitted.
        """
        with self._lock:
            self._expire_pending(time.monotonic())
            delivery = seq if address is not None else self._inflight.get(seq, 0)
            dispatch = self._pending.get(delivery)
            if dispatch is None or (address is not None and address != dispatch.address):
                return
            del self._pending[delivery]
            del self._inflight[dispatch.upstream_sequence]
            self._stats[dispatch.target_id].acked += 1
            self._total_acked += 1

    @property
    def pending_count(self) -> int:
        """Return the number of dispatched packets awaiting ACKs."""
        with self._lock:
            self._expire_pending(time.monotonic())
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
            self._expire_pending(time.monotonic())
            s = self._stats.get(neuron_id)
            return RouteStats(s.dispatched, s.acked, s.dropped) if s else None
