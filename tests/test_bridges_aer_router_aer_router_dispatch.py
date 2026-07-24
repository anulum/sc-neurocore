# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAERRouterDispatch from former test_bridges_aer_router.py

"""Focused suite: TestAERRouterDispatch from former test_bridges_aer_router.py."""

from __future__ import annotations

from tests.bridges_aer_router_support import *  # noqa: F403


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
