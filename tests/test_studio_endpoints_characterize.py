# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints characterize

"""Focused suite: TestCharacterizeEndpoint from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403

class TestCharacterizeEndpoint:
    def test_characterize_model(self, client):
        r = client.post(
            "/api/characterize",
            json={
                "name": MODEL,
                "dt": 0.5,
                "duration": 20.0,
                "current": 10.0,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "pattern" in data
        assert "fi_curve" in data
        assert "top_sensitivities" in data

