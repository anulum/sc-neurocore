# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints bifurcation

"""Focused suite: TestBifurcationEndpoint from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403

class TestBifurcationEndpoint:
    def test_bifurcation_model(self, client):
        r = client.post(
            "/api/bifurcation",
            json={
                "model_name": MODEL,
                "duration": 20.0,
                "current": 10.0,
                "params": {"v_rest": -65.0},
                "sweep_param": "v_rest",
                "sweep_min": -75,
                "sweep_max": -55,
                "sweep_steps": 5,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "param_values" in data
        assert "attractors" in data

