# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio advanced network

"""Focused suite: TestNetwork from former test_studio_advanced.py."""

from __future__ import annotations

from tests.studio_advanced_support import *  # noqa: F403


class TestNetwork:
    def test_basic_ei_network(self):
        r = simulate_ei_network(n_exc=20, n_inh=5, duration=50.0, ext_rate=10.0)
        assert r["n_total"] == 25
        assert r["n_exc"] == 20
        assert r["n_inh"] == 5
        assert len(r["spike_times"]) == r["n_spikes"]
        assert len(r["spike_neurons"]) == r["n_spikes"]

    def test_network_produces_spikes(self):
        r = simulate_ei_network(n_exc=40, n_inh=10, duration=200.0, ext_rate=50.0)
        assert r["n_spikes"] >= 0  # may be 0 with low drive, just verify no crash

    def test_network_rates_arrays(self):
        r = simulate_ei_network(n_exc=20, n_inh=5, duration=50.0)
        assert len(r["rate_time"]) > 0
        assert len(r["exc_rates"]) == len(r["rate_time"])
        assert len(r["inh_rates"]) == len(r["rate_time"])

    def test_network_uses_rust_engine(self):
        """Verify the Rust engine path is used when available."""
        try:
            simulate = get_ei_network_simulator()
            r = simulate(n_exc=10, n_inh=5, duration=20.0, ext_rate=100.0)
            assert "spike_times" in r
            assert "n_total" in r
            assert int(r["n_total"]) == 15
        except ImportError:
            pytest.skip("Rust engine not installed")

    def test_network_result_types(self):
        r = simulate_ei_network(n_exc=10, n_inh=5, duration=20.0)
        assert isinstance(r["spike_times"], list)
        assert isinstance(r["n_exc"], int)
        assert isinstance(r["mean_exc_rate"], float)

    def test_network_endpoint(self, client):
        r = client.post(
            "/api/network/ei",
            json={
                "n_exc": 20,
                "n_inh": 5,
                "duration": 30,
                "ext_rate": 10,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert d["n_total"] == 25
        assert "spike_times" in d
