# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints sensitivity

"""Focused suite: TestSensitivityEndpoint from former test_studio_endpoints.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.testclient import TestClient

from tests.studio_endpoints_support import (
    MODEL,
)
from tests.studio_endpoints_support import client as client


class TestSensitivityEndpoint:
    """The Studio sensitivity endpoint."""

    def test_sensitivity_model(self, client: TestClient) -> None:
        """A model sensitivity sweep returns the base rate and sensitivities."""
        r = client.post(
            "/api/sensitivity",
            json={
                "model_name": MODEL,
                "duration": 20.0,
                "current": 10.0,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "base_rate" in data
        assert "sensitivities" in data
