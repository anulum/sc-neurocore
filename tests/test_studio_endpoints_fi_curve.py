# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints fi curve

"""Focused suite: TestFICurveEndpoint from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403

class TestFICurveEndpoint:
    def test_fi_curve_model(self, client):
        r = client.post(
            "/api/fi-curve",
            json={
                "model_name": MODEL,
                "duration": 30.0,
                "i_min": 0,
                "i_max": 20,
                "i_steps": 3,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "currents" in data
        assert "rates" in data
        assert len(data["currents"]) == 3

