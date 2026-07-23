# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio advanced errors

"""Focused suite: TestErrorHandling from former test_studio_advanced.py."""

from __future__ import annotations

from tests.studio_advanced_support import *  # noqa: F403

class TestErrorHandling:
    def test_bad_model_name(self, client):
        r = client.post(
            "/api/models/simulate",
            json={
                "name": "NonExistentNeuron",
                "current": 10,
                "duration": 50,
            },
        )
        assert r.status_code == 422

    def test_bad_ode_equation(self, client):
        r = client.post(
            "/api/simulate",
            json={
                "equations": ["this is not an ODE"],
                "dt": 0.1,
                "duration": 10,
            },
        )
        assert r.status_code == 422

    def test_negative_duration(self, client):
        r = client.post(
            "/api/simulate",
            json={
                "equations": ["dv/dt = I"],
                "dt": 0.1,
                "duration": -10,
            },
        )
        assert r.status_code == 422

