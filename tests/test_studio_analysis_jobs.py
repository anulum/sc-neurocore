# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio analysis job route tests

"""Route-level tests for asynchronous heavy-analysis job submission."""

from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.policy_routes import (
    build_default_studio_route_policy_registry,
)


def test_analysis_and_model_scan_job_routes_are_registered_in_route_policy() -> None:
    """New job routes must be present in the default policy inventory."""

    registry = build_default_studio_route_policy_registry()
    scan_job = registry.policy_for("POST", "/api/models/scan/jobs")
    analysis_job = registry.policy_for("POST", "/api/analysis/jobs")
    assert scan_job.audit_action == "studio.models.scan.job"
    assert analysis_job.audit_action == "studio.analysis.job"


def test_analysis_job_route_returns_async_job_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/analysis/jobs must not block the HTTP thread waiting for work."""

    from sc_neurocore.studio.api import simulation as simulation_routes

    def _fake_bifurcation(
        simulate_fn: Any,
        base_cfg: dict[str, Any],
        sweep_param: str,
        sweep_min: float,
        sweep_max: float,
        sweep_steps: int,
    ) -> dict[str, object]:
        return {
            "sweep_param": sweep_param,
            "values": [sweep_min, sweep_max],
            "rates": [0.0, 1.0],
            "steps": sweep_steps,
        }

    monkeypatch.setattr(simulation_routes, "bifurcation_sweep", _fake_bifurcation)
    client = TestClient(
        create_app(StudioRuntimeSettings()),
        base_url="http://127.0.0.1",
    )
    response = client.post(
        "/api/analysis/jobs",
        json={
            "analysis": "bifurcation",
            "payload": {
                "model_name": "LIFNeuron",
                "sweep_param": "tau_m",
                "sweep_min": 5.0,
                "sweep_max": 15.0,
                "sweep_steps": 5,
                "duration": 50.0,
                "dt": 0.1,
                "current": 10.0,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_mode"] == "async_job"
    assert payload["schema_version"] == "studio.analysis.job.v1"
    assert payload["analysis"] == "bifurcation"
    assert str(payload["job_id"]).startswith("sj_")
    assert payload["status_route"] == f"/api/studio/jobs/{payload['job_id']}"
    assert payload["job"]["kind"] == "analysis"


def test_over_budget_heatmap_recommends_existing_analysis_jobs_route() -> None:
    """Heavy analysis budget rejections must advertise a live job route."""

    client = TestClient(
        create_app(StudioRuntimeSettings(max_sync_analysis_total_steps=10)),
        base_url="http://127.0.0.1",
    )
    response = client.post(
        "/api/heatmap",
        json={
            "model_name": "LIFNeuron",
            "param_x": "tau_m",
            "x_min": 1.0,
            "x_max": 10.0,
            "x_steps": 10,
            "param_y": "R",
            "y_min": 1.0,
            "y_max": 10.0,
            "y_steps": 10,
            "duration": 100.0,
            "dt": 0.1,
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["execution_mode"] == "job_required"
    assert detail["recommended_route"] == "POST /api/analysis/jobs"
    assert detail["async_required"] is True
