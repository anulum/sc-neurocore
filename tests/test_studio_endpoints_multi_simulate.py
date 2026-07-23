# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints multi simulate

"""Focused suite: TestMultiSimulate from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403

class TestMultiSimulate:
    def test_multi_simulate(self, client):
        r = client.post(
            "/api/multi-simulate",
            json=[
                {"name": MODEL, "duration": 20, "current": 10},
                {"name": "ChayNeuron", "duration": 20, "current": 10},
            ],
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert all("time" in d for d in data)

    def test_empty_multi_simulate_returns_empty_result(self, client):
        """An empty bounded batch is a valid no-op."""

        response = client.post("/api/multi-simulate", json=[])

        assert response.status_code == 200
        assert response.json() == []

