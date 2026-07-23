# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints network

"""Focused suite: TestNetworkEndpoint from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403

class TestNetworkEndpoint:
    def test_network_default(self, client):
        r = client.post("/api/network/ei", json={})
        assert r.status_code == 200
        data = r.json()
        assert "spike_times" in data
        assert "spike_neurons" in data
        assert "rate_time" in data
        assert data["n_total"] == 100

    def test_network_custom(self, client):
        r = client.post(
            "/api/network/ei",
            json={
                "n_exc": 40,
                "n_inh": 10,
                "duration": 50.0,
                "ext_rate": 20.0,
                "w_ee": 0.05,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["n_exc"] == 40
        assert data["n_inh"] == 10
        assert data["n_total"] == 50

