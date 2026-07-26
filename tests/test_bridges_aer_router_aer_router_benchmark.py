# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAERRouterBenchmark from former test_bridges_aer_router.py

"""Focused suite: TestAERRouterBenchmark from former test_bridges_aer_router.py."""

from __future__ import annotations

from tests.bridges_aer_router_support import *  # noqa: F403


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
        """Register 10,000 routes within the local or instrumented-CI budget."""
        router = AERRouter()
        t0 = time.perf_counter()
        for i in range(10_000):
            router.register_route(neuron_id=i, addr=f"h:{5000 + i}")
        elapsed = time.perf_counter() - t0
        assert router.route_count == 10_000
        max_elapsed = 2.0 if os.environ.get("CI") else 0.5
        assert elapsed < max_elapsed, f"10k registrations took {elapsed:.2f}s"
