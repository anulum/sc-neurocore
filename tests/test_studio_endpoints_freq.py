# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints freq

"""Focused suite: TestFreqResponseEndpoint from former test_studio_endpoints.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.testclient import TestClient

from tests.studio_endpoints_support import (
    MODEL,
)
from tests.studio_endpoints_support import client as client


class TestFreqResponseEndpoint:
    """The Studio frequency-response endpoint."""

    def test_freq_response(self, client: TestClient) -> None:
        """A frequency response returns frequencies and rates for the given bias and depth."""
        r = client.post(
            "/api/freq-response",
            json={
                "model_name": MODEL,
                "duration": 20.0,
                "bias": 10.0,
                "freq_min": 1,
                "freq_max": 50,
                "n_freqs": 3,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "frequencies_hz" in data
        assert "rates" in data
        assert data["bias"] == 10.0 and data["depth"] == 0.5

    def test_the_retired_amplitude_is_refused_with_its_reason(self, client: TestClient) -> None:
        """The retired amplitude field is refused with the reason to send bias and depth."""
        # "amplitude" meant a mean-zero sine; reading it as the new bias would
        # answer a different question under the old name.
        r = client.post(
            "/api/freq-response",
            json={"model_name": MODEL, "duration": 20.0, "amplitude": 10, "n_freqs": 3},
        )
        assert r.status_code == 422
        assert "send bias and depth" in r.text

    def test_a_zero_bias_is_refused(self, client: TestClient) -> None:
        """A zero bias is refused with 422."""
        r = client.post(
            "/api/freq-response",
            json={"model_name": MODEL, "duration": 20.0, "bias": 0.0, "n_freqs": 3},
        )
        assert r.status_code == 422
        assert "non-zero" in r.text
