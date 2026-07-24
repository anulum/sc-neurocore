# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio rust integration ei network

"""Focused suite: TestRustEINetwork from former test_studio_rust_integration.py."""

from __future__ import annotations

from tests.studio_rust_integration_support import *  # noqa: F403


class TestRustEINetwork:
    def test_rust_ei_network_direct(self):
        _bridge_engine()
        simulate = get_ei_network_simulator()
        r = simulate(n_exc=20, n_inh=5, duration=50.0, ext_rate=100.0)
        assert int(r["n_total"]) == 25
        assert int(r["n_exc"]) == 20
        assert "spike_times" in r
        assert "exc_rates" in r

    def test_rust_ei_network_via_python(self):
        _bridge_engine()
        from sc_neurocore.studio.network import simulate_ei_network

        r = simulate_ei_network(n_exc=20, n_inh=5, duration=50.0, ext_rate=100.0)
        assert r["n_total"] == 25
        assert isinstance(r["spike_times"], list)
        assert isinstance(r["mean_exc_rate"], (int, float))

    def test_rust_ei_result_types(self):
        sce = _inner_engine()
        r = sce.py_simulate_ei_network(n_exc=10, n_inh=5, duration=20.0)
        assert hasattr(r["spike_times"], "tolist")
        assert hasattr(r["rate_time"], "tolist")

    def test_numpy_fallback_works(self):
        from sc_neurocore.studio.network import _simulate_numpy

        r = _simulate_numpy(
            n_exc=10,
            n_inh=5,
            w_ee=0.1,
            w_ei=0.4,
            w_ie=0.1,
            w_ii=0.4,
            p_conn=0.2,
            ext_rate=10.0,
            duration=20.0,
            dt=0.1,
        )
        assert r["n_total"] == 15
        assert isinstance(r["spike_times"], list)
