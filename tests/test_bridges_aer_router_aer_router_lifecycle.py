# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAERRouterLifecycle from former test_bridges_aer_router.py

"""Focused suite: TestAERRouterLifecycle from former test_bridges_aer_router.py."""

from __future__ import annotations

from tests.bridges_aer_router_support import *  # noqa: F403


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
