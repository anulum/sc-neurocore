# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAERRouterBenchmark from former test_bridges_aer_router.py

"""Focused suite: TestAERRouterBenchmark from former test_bridges_aer_router.py."""

from __future__ import annotations

import os
import time

from sc_neurocore.bridges.aer_router import AERRouter, SpikePacket
from tests.performance_guard import assert_load_tolerant_throughput


class TestAERRouterBenchmark:
    """Check throughput for acknowledged dispatch, packet encoding and route registration."""

    def test_dispatch_throughput_10k(self) -> None:
        """Complete and acknowledge ten thousand dispatches within the throughput budget."""
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        pkt = SpikePacket(source_id=0, target_id=1, timestamp=0, spike_len=1, sequence=0)
        t0 = time.perf_counter()
        for i in range(10_000):
            pkt.sequence = i
            assert router.dispatch_spike(pkt)
            router.ack_received(i)
        elapsed = time.perf_counter() - t0
        assert router.total_sent == router.total_acked == 10_000
        assert router.pending_count == 0
        throughput = 10_000 / max(elapsed, 1e-9)
        assert_load_tolerant_throughput(
            label="AER dispatch and ACK",
            observed_per_second=throughput,
            strict_minimum_per_second=10_000.0,
        )

    def test_encode_decode_throughput_100k(self) -> None:
        """100,000 encode+decode cycles must complete in < 2 seconds."""
        pkt = SpikePacket(source_id=42, target_id=99, timestamp=1000, spike_len=4, sequence=1)
        t0 = time.perf_counter()
        for _ in range(100_000):
            raw = pkt.encode()
            SpikePacket.decode(raw)
        elapsed = time.perf_counter() - t0
        throughput = 100_000 / max(elapsed, 1e-9)
        assert_load_tolerant_throughput(
            label="AER encode/decode",
            observed_per_second=throughput,
            strict_minimum_per_second=50_000.0,
        )

    def test_route_registration_throughput(self) -> None:
        """Register 10,000 routes within the local or instrumented-CI budget."""
        router = AERRouter()
        t0 = time.perf_counter()
        for i in range(10_000):
            router.register_route(neuron_id=i, addr=f"h:{5000 + i}")
        elapsed = time.perf_counter() - t0
        assert router.route_count == 10_000
        max_elapsed = 2.0 if os.environ.get("CI") else 0.5
        assert_load_tolerant_throughput(
            label="AER route registration",
            observed_per_second=10_000 / elapsed,
            strict_minimum_per_second=10_000 / max_elapsed,
        )
