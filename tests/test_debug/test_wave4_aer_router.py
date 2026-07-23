# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAERRouter from former test_wave4.py

"""Focused suite: TestAERRouter from former test_wave4.py."""

from __future__ import annotations

from wave4_support import *  # noqa: F403

class TestAERRouter:
    def test_register(self):
        r = AERRouter()
        r.register_route(100, "127.0.0.1:9001")
        assert r.route_count == 1

    def test_unregister(self):
        r = AERRouter()
        r.register_route(100, "127.0.0.1:9001")
        r.unregister_route(100)
        assert r.route_count == 0

    def test_dispatch_unregistered(self):
        r = AERRouter()
        assert not r.dispatch_spike(SpikePacket(target_id=999))

    def test_dispatch_increments_stats(self):
        r = AERRouter()
        r.register_route(100, "127.0.0.1:9001")
        r.dispatch_spike(SpikePacket(target_id=100, sequence=1))
        assert r.total_sent == 1
        s = r.get_stats(100)
        assert s.dispatched == 1

    def test_ack_clears_pending(self):
        r = AERRouter()
        r.register_route(100, "127.0.0.1:9001")
        r.dispatch_spike(SpikePacket(target_id=100, sequence=42))
        assert r.pending_count == 1
        r.ack_received(42)
        assert r.pending_count == 0

    def test_multi_route(self):
        r = AERRouter()
        for nid, port in [(10, 9001), (20, 9002), (30, 9003)]:
            r.register_route(nid, f"127.0.0.1:{port}")
        assert r.route_count == 3
        for nid in [10, 20, 30]:
            assert r.dispatch_spike(SpikePacket(target_id=nid, sequence=nid))
        assert r.total_sent == 3
