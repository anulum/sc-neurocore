# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAERRouterDispatch from former test_bridges_aer_router.py

"""Focused suite: TestAERRouterDispatch from former test_bridges_aer_router.py."""

from __future__ import annotations

import time

from sc_neurocore.bridges.aer_router import AERRouter, SpikePacket


class TestAERRouterDispatch:
    """Dispatch, ACK, pending tracking."""

    def _make_pkt(self, target: int = 1, seq: int = 0) -> SpikePacket:
        """Construct a packet addressed to the chosen route."""
        return SpikePacket(source_id=0, target_id=target, timestamp=0, spike_len=1, sequence=seq)

    def test_dispatch_to_registered_target_succeeds(self) -> None:
        """Accept dispatch when the target route is registered."""
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        ok = router.dispatch_spike(self._make_pkt(target=1, seq=0))
        assert ok is True
        assert router.total_sent == 1

    def test_dispatch_to_unregistered_target_fails(self) -> None:
        """Refuse unrouted events without changing the sent count."""
        router = AERRouter()
        ok = router.dispatch_spike(self._make_pkt(target=99, seq=0))
        assert ok is False
        assert router.total_sent == 0

    def test_dispatch_increments_pending(self) -> None:
        """A dispatched sequence awaits acknowledgement."""
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        router.dispatch_spike(self._make_pkt(target=1, seq=100))
        assert router.pending_count >= 1

    def test_ack_clears_pending(self) -> None:
        """A repeated ACK clears and counts the matching packet once."""
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        router.dispatch_spike(self._make_pkt(target=1, seq=42))
        router.ack_received(seq=42)
        router.ack_received(seq=42)
        assert router.pending_count == 0
        assert router.total_acked == 1
        stats = router.get_stats(neuron_id=1)
        assert stats is not None
        assert stats.acked == 1

    def test_multiple_dispatches(self) -> None:
        """Count each accepted dispatch across distinct sequences."""
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        for i in range(100):
            router.dispatch_spike(self._make_pkt(target=1, seq=i))
        assert router.total_sent == 100

    def test_ack_for_unknown_seq_is_safe(self) -> None:
        """An ACK without a pending dispatch leaves all counters unchanged."""
        router = AERRouter()
        router.ack_received(seq=999)
        assert router.total_acked == 0

    def test_dispatch_updates_per_route_stats(self) -> None:
        """Attribute each dispatch to its registered destination."""
        router = AERRouter()
        router.register_route(neuron_id=5, addr="h:5000")
        for i in range(10):
            router.dispatch_spike(self._make_pkt(target=5, seq=i))
        stats = router.get_stats(neuron_id=5)
        assert stats is not None
        assert stats.dispatched == 10

    def test_inflight_sequence_and_endpoint_preserve_route_attribution(self) -> None:
        """A pending sequence stays with its original destination until a valid ACK."""
        router = AERRouter()
        router.register_route(1, "a:5000")
        router.register_route(2, "b:5000")
        delivery = router.dispatch_for_delivery(self._make_pkt(target=1, seq=9))
        assert delivery is not None and delivery.sequence > 0
        assert router.dispatch_for_delivery(self._make_pkt(target=2, seq=9)) is None
        router.ack_received(delivery.sequence, address="b:5000")
        assert router.pending_count == 1
        assert router.total_acked == 0
        router.ack_received(delivery.sequence, address="a:5000")
        assert router.pending_count == 0
        first, second = router.get_stats(1), router.get_stats(2)
        assert first is not None and first.acked == 1
        assert second is not None and second.acked == 0
        assert router.dispatch_spike(self._make_pkt(target=2, seq=9))

    def test_missing_acknowledgements_are_bounded_and_expire(self) -> None:
        """Pending entries account for loss and make room for fresh spikes."""
        router = AERRouter()
        router.register_route(1, "a:5000")
        limit = 4096
        for sequence in range(limit):
            assert router.dispatch_spike(self._make_pkt(seq=sequence))
        assert not router.dispatch_spike(self._make_pkt(seq=limit))
        time.sleep(2.0)
        router.ack_received(0)
        assert router.pending_count == 0
        assert router.total_acked == 0
        route = router.get_stats(1)
        assert route is not None and route.dropped == limit + 1
        assert router.dispatch_spike(self._make_pkt(seq=0))

    def test_delayed_ack_cannot_complete_reused_sequence_at_shared_endpoint(self) -> None:
        """An expired delivery token carries no credit into a later route."""
        router = AERRouter()
        for route in (1, 2):
            router.register_route(route, "shared:5000")
        first = router.dispatch_for_delivery(self._make_pkt(target=1, seq=25))
        assert first is not None
        time.sleep(2.0)
        assert router.pending_count == 0
        second = router.dispatch_for_delivery(self._make_pkt(target=2, seq=25))
        assert second is not None and second.sequence != first.sequence
        router.ack_received(first.sequence, address="shared:5000")
        assert router.total_acked == 0 and router.pending_count == 1
        router.ack_received(second.sequence, address="shared:5000")
        route_one, route_two = router.get_stats(1), router.get_stats(2)
        assert router.total_acked == 1 and router.pending_count == 0
        assert route_one is not None and route_one.dropped == 1 and route_one.acked == 0
        assert route_two is not None and route_two.acked == 1

    def test_delayed_ack_cannot_complete_delivery_on_replacement_router(self) -> None:
        """A replacement router assigns a fresh wire sequence at the same endpoint."""
        previous, replacement = AERRouter(), AERRouter()
        packet = self._make_pkt(seq=-25)
        for router in (previous, replacement):
            router.register_route(packet.target_id, "a:5000")
        old = previous.dispatch_for_delivery(packet)
        current = replacement.dispatch_for_delivery(packet)
        assert old is not None and current is not None
        assert old.sequence != current.sequence
        replacement.ack_received(old.sequence, address="a:5000")
        assert replacement.pending_count == 1
        assert replacement.total_acked == 0
        replacement.ack_received(current.sequence, address="a:5000")
        assert replacement.pending_count == 0
        assert replacement.total_acked == 1

    def test_delivery_sequence_does_not_wrap_after_exhaustion(self) -> None:
        """The last signed wire token remains usable before admission stops."""
        router = AERRouter()
        router.register_route(1, "a:5000")
        router._delivery_base = (1 << 63) - 2
        router._next_delivery = (1 << 63) - 2
        last = router.dispatch_for_delivery(self._make_pkt(seq=1))
        assert last is not None and last.sequence == (1 << 63) - 2
        assert SpikePacket.decode(last.encode()) == last
        router.ack_received(last.sequence, address="a:5000")
        assert router.dispatch_for_delivery(self._make_pkt(seq=2)) is None
        assert router.total_sent == router.total_acked == 1
