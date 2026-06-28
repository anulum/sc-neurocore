# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio nullcline analysis budget route tests

"""Route-level tests for Studio nullcline grid budget enforcement."""

from __future__ import annotations

from typing import Any

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings


def test_nullclines_rejected_over_grid_budget_before_analysis_runs() -> None:
    """Oversized nullcline grids fail at the API budget guard with path-free detail."""

    client = TestClient(
        create_app(StudioRuntimeSettings(max_sync_analysis_simulations=3_000)),
        base_url="http://127.0.0.1",
    )

    response = client.post(
        "/api/nullclines",
        json={
            "equations": ["dv/dt = w", "dw/dt = -v"],
            "params": {},
            "var_names": ["v", "w"],
            "ranges": {"v": [-1.0, 1.0], "w": [-1.0, 1.0]},
            "grid_size": 60,
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, dict)
    typed_detail = dict[str, Any](detail)
    assert typed_detail["limit"] == "simulations"
    assert typed_detail["projected"] == 3_600
    assert typed_detail["allowed"] == 3_000
    reason = typed_detail["reason"]
    assert isinstance(reason, str)
    assert "/home/" not in reason
    assert "/media/" not in reason


def test_nullclines_within_grid_budget_returns_analysis_metadata() -> None:
    """A bounded nullcline grid still uses the production analysis route."""

    client = TestClient(
        create_app(StudioRuntimeSettings(max_sync_analysis_simulations=500)),
        base_url="http://127.0.0.1",
    )

    response = client.post(
        "/api/nullclines",
        json={
            "equations": ["dv/dt = w", "dw/dt = -v"],
            "params": {},
            "var_names": ["v", "w"],
            "ranges": {"v": [-1.0, 1.0], "w": [-1.0, 1.0]},
            "grid_size": 20,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_metadata"]["analysis_type"] == "nullclines"
    assert payload["analysis_metadata"]["evidence_classification"] == "analysis"
