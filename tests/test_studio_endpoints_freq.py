# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints freq

"""Focused suite: TestFreqResponseEndpoint from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403


class TestFreqResponseEndpoint:
    def test_freq_response(self, client):
        r = client.post(
            "/api/freq-response",
            json={
                "model_name": MODEL,
                "duration": 20.0,
                "current": 10.0,
                "amplitude": 10,
                "freq_min": 1,
                "freq_max": 50,
                "n_freqs": 3,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "frequencies_hz" in data
        assert "rates" in data
