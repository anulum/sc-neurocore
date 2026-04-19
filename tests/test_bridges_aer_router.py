# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for bridges.aer_router

from __future__ import annotations

import struct
import time

import pytest

from sc_neurocore.bridges.aer_router import (
    AERRouter,
    SpikePacket,
    RouteStats,
    PACKET_SIZE,
)


# ---------------------------------------------------------------------------
# SpikePacket encode / decode
# ---------------------------------------------------------------------------


class TestSpikePacket:
    """Packet encode / decode round-trip and edge cases."""

    def test_encode_decode_basic(self):
        pkt = SpikePacket(source_id=42, target_id=99, timestamp=1000, spike_len=4, sequence=1)
        raw = pkt.encode()
        assert len(raw) == PACKET_SIZE
        restored = SpikePacket.decode(raw)
        assert restored.source_id == 42
        assert restored.target_id == 99
        assert restored.timestamp == 1000
        assert restored.spike_len == 4
        assert restored.sequence == 1

    def test_encode_decode_zero_fields(self):
        pkt = SpikePacket()
        restored = SpikePacket.decode(pkt.encode())
        assert restored.source_id == 0
        assert restored.target_id == 0
        assert restored.timestamp == 0
        assert restored.spike_len == 0
        assert restored.sequence == 0

    def test_encode_decode_negative_sequence(self):
        pkt = SpikePacket(source_id=1, target_id=2, timestamp=5, spike_len=1, sequence=-42)
        restored = SpikePacket.decode(pkt.encode())
        assert restored.sequence == -42

    def test_encode_decode_large_timestamp(self):
        ts = 2**48
        pkt = SpikePacket(source_id=10, target_id=20, timestamp=ts, spike_len=8, sequence=100)
        restored = SpikePacket.decode(pkt.encode())
        assert restored.timestamp == ts

    def test_encode_is_big_endian(self):
        pkt = SpikePacket(source_id=1, target_id=0, timestamp=0, spike_len=0, sequence=0)
        raw = pkt.encode()
        assert raw[3] == 1  # big-endian u32: least significant byte last

    @pytest.mark.parametrize(
        "src,tgt,seq",
        [
            (0, 0, 0),
            (1, 2, 3),
            (2**32 - 1, 2**32 - 1, 2**63 - 1),
            (100, 200, -100),
        ],
    )
    def test_fuzz_encode_decode(self, src, tgt, seq):
        pkt = SpikePacket(source_id=src, target_id=tgt, timestamp=0, spike_len=0, sequence=seq)
        restored = SpikePacket.decode(pkt.encode())
        assert restored.source_id == src
        assert restored.target_id == tgt
        assert restored.sequence == seq

    def test_decode_ignores_trailing_data(self):
        pkt = SpikePacket(source_id=5, target_id=10, timestamp=99, spike_len=1, sequence=7)
        raw = pkt.encode() + b"\xff" * 16
        restored = SpikePacket.decode(raw)
        assert restored.source_id == 5
        assert restored.sequence == 7

    def test_decode_short_data_raises(self):
        with pytest.raises(struct.error):
            SpikePacket.decode(b"\x00" * 4)

    def test_packet_size_constant(self):
        assert PACKET_SIZE == 28


# ---------------------------------------------------------------------------
# AERRouter lifecycle
# ---------------------------------------------------------------------------


class TestAERRouterLifecycle:
    """Route registration, unregistration, counting."""

    def test_empty_router(self):
        router = AERRouter()
        assert router.route_count == 0
        assert router.total_sent == 0
        assert router.total_acked == 0
        assert router.pending_count == 0

    def test_register_single_route(self):
        router = AERRouter()
        router.register_route(neuron_id=1, addr="192.168.1.1:5000")
        assert router.route_count == 1

    def test_register_multiple_routes(self):
        router = AERRouter()
        for i in range(20):
            router.register_route(neuron_id=i, addr=f"host{i}:5000")
        assert router.route_count == 20

    def test_unregister_decreases_count(self):
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        router.register_route(neuron_id=2, addr="h:5001")
        assert router.route_count == 2
        router.unregister_route(neuron_id=1)
        assert router.route_count == 1

    def test_unregister_nonexistent_is_noop(self):
        router = AERRouter()
        router.unregister_route(neuron_id=999)
        assert router.route_count == 0

    def test_re_register_overwrites(self):
        router = AERRouter()
        router.register_route(neuron_id=1, addr="old:5000")
        router.register_route(neuron_id=1, addr="new:5001")
        assert router.route_count == 1


# ---------------------------------------------------------------------------
# Spike dispatch and ACK
# ---------------------------------------------------------------------------


class TestAERRouterDispatch:
    """Dispatch, ACK, pending tracking."""

    def _make_pkt(self, target=1, seq=0):
        return SpikePacket(source_id=0, target_id=target, timestamp=0, spike_len=1, sequence=seq)

    def test_dispatch_to_registered_target_succeeds(self):
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        ok = router.dispatch_spike(self._make_pkt(target=1, seq=0))
        assert ok is True
        assert router.total_sent == 1

    def test_dispatch_to_unregistered_target_fails(self):
        router = AERRouter()
        ok = router.dispatch_spike(self._make_pkt(target=99, seq=0))
        assert ok is False
        assert router.total_sent == 0

    def test_dispatch_increments_pending(self):
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        router.dispatch_spike(self._make_pkt(target=1, seq=100))
        assert router.pending_count >= 1

    def test_ack_clears_pending(self):
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        router.dispatch_spike(self._make_pkt(target=1, seq=42))
        router.ack_received(seq=42)
        assert router.total_acked == 1

    def test_multiple_dispatches(self):
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        for i in range(100):
            router.dispatch_spike(self._make_pkt(target=1, seq=i))
        assert router.total_sent == 100

    def test_ack_for_unknown_seq_is_safe(self):
        router = AERRouter()
        router.ack_received(seq=999)
        assert router.total_acked == 1

    def test_dispatch_updates_per_route_stats(self):
        router = AERRouter()
        router.register_route(neuron_id=5, addr="h:5000")
        for i in range(10):
            router.dispatch_spike(self._make_pkt(target=5, seq=i))
        stats = router.get_stats(neuron_id=5)
        assert stats is not None
        assert stats.dispatched == 10


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestAERRouterStats:
    """RouteStats correctness."""

    def test_get_stats_unknown_neuron_returns_none(self):
        router = AERRouter()
        assert router.get_stats(neuron_id=999) is None

    def test_stats_are_copies(self):
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        s1 = router.get_stats(neuron_id=1)
        s2 = router.get_stats(neuron_id=1)
        assert s1 is not s2

    def test_fresh_stats_are_zero(self):
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        stats = router.get_stats(neuron_id=1)
        assert stats.dispatched == 0
        assert stats.acked == 0
        assert stats.dropped == 0

    def test_route_stats_dataclass(self):
        rs = RouteStats(dispatched=10, acked=8, dropped=2)
        assert rs.dispatched == 10
        assert rs.acked == 8
        assert rs.dropped == 2


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestAERRouterBenchmark:
    """Performance checks."""

    def test_dispatch_throughput_10k(self):
        """10,000 dispatches must complete in < 1 second."""
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        pkt = SpikePacket(source_id=0, target_id=1, timestamp=0, spike_len=1, sequence=0)
        t0 = time.perf_counter()
        for i in range(10_000):
            pkt.sequence = i
            router.dispatch_spike(pkt)
        elapsed = time.perf_counter() - t0
        throughput = 10_000 / max(elapsed, 1e-9)
        assert elapsed < 1.0, f"10k dispatches took {elapsed:.2f}s ({throughput:.0f}/s)"

    def test_encode_decode_throughput_100k(self):
        """100,000 encode+decode cycles must complete in < 2 seconds."""
        pkt = SpikePacket(source_id=42, target_id=99, timestamp=1000, spike_len=4, sequence=1)
        t0 = time.perf_counter()
        for i in range(100_000):
            raw = pkt.encode()
            SpikePacket.decode(raw)
        elapsed = time.perf_counter() - t0
        throughput = 100_000 / max(elapsed, 1e-9)
        assert elapsed < 2.0, f"100k encode/decode took {elapsed:.2f}s ({throughput:.0f}/s)"

    def test_route_registration_throughput(self):
        """Register 10,000 routes in < 0.5 seconds."""
        router = AERRouter()
        t0 = time.perf_counter()
        for i in range(10_000):
            router.register_route(neuron_id=i, addr=f"h:{5000 + i}")
        elapsed = time.perf_counter() - t0
        assert router.route_count == 10_000
        assert elapsed < 0.5, f"10k registrations took {elapsed:.2f}s"
