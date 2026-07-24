# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio free tests from test_studio_endpoints.py

"""Module-level tests extracted from test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403


def test_import_trace_rejects_empty_voltage(client: TestClient) -> None:
    """Trace import requires a non-empty voltage vector."""

    response = client.post("/api/import-trace", json={"voltage": [], "dt": 0.1})

    assert response.status_code == 422
    assert response.json()["detail"] == "Expected {voltage: [...], dt: float}"
