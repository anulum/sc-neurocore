# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio model-scan analysis budget route tests

"""Route-level tests for Studio model-scan budget enforcement."""

from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.api import catalogue as catalogue_routes
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings


def test_model_scan_rejected_over_catalogue_budget_before_scan_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized model catalogues fail at the route guard with path-free detail."""

    monkeypatch.setattr(
        catalogue_routes,
        "list_models",
        lambda: [
            {"name": "ModelA", "category": "CatA"},
            {"name": "ModelB", "category": "CatB"},
            {"name": "ModelC", "category": "CatC"},
        ],
    )

    def _scan_all_models(*, current: float, duration: float) -> dict[str, object]:
        raise AssertionError("scan_all_models must not run after budget rejection")

    monkeypatch.setattr(catalogue_routes, "scan_all_models", _scan_all_models)
    client = TestClient(
        create_app(StudioRuntimeSettings(max_sync_analysis_simulations=2)),
        base_url="http://127.0.0.1",
    )

    response = client.get("/api/models/scan")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, dict)
    typed_detail = dict[str, Any](detail)
    assert typed_detail["limit"] == "simulations"
    assert typed_detail["projected"] == 3
    assert typed_detail["allowed"] == 2
    assert typed_detail["execution_mode"] == "job_required"
    assert typed_detail["async_required"] is True
    assert typed_detail["recommended_route"] == "POST /api/models/scan/jobs"
    assert typed_detail["schema_version"] == "studio.model-scan.v1"
    assert typed_detail["evidence_classification"] == "analysis"
    reason = typed_detail["reason"]
    assert isinstance(reason, str)
    assert "/home/" not in reason
    assert "/media/" not in reason


def test_model_scan_within_catalogue_budget_runs_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded model catalogue still uses the production scan route."""

    monkeypatch.setattr(
        catalogue_routes,
        "list_models",
        lambda: [{"name": "ModelA", "category": "CatA"}],
    )

    def _scan_all_models(*, current: float, duration: float) -> dict[str, object]:
        return {
            "models": [],
            "scan_metadata": {
                "current": current,
                "duration": duration,
                "evidence_classification": "analysis",
                "schema_version": "studio.model-scan.v1",
                "status": "completed",
            },
            "schema_version": "studio.model-scan.v1",
        }

    monkeypatch.setattr(catalogue_routes, "scan_all_models", _scan_all_models)
    client = TestClient(
        create_app(StudioRuntimeSettings(max_sync_analysis_simulations=1)),
        base_url="http://127.0.0.1",
    )

    response = client.get("/api/models/scan")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "studio.model-scan.v1"
    assert payload["scan_metadata"]["evidence_classification"] == "analysis"


def test_model_scan_job_route_returns_async_job_without_blocking_on_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized catalogues can be submitted as model_scan jobs immediately."""

    monkeypatch.setattr(
        catalogue_routes,
        "list_models",
        lambda: [
            {"name": "ModelA", "category": "CatA"},
            {"name": "ModelB", "category": "CatB"},
            {"name": "ModelC", "category": "CatC"},
        ],
    )

    def _scan_all_models(*, current: float, duration: float) -> dict[str, object]:
        return {
            "models": [{"name": "ModelA", "pattern": "tonic"}],
            "scan_metadata": {
                "current": current,
                "duration": duration,
                "evidence_classification": "analysis",
                "schema_version": "studio.model-scan.v1",
                "status": "completed",
            },
            "schema_version": "studio.model-scan.v1",
        }

    monkeypatch.setattr(catalogue_routes, "scan_all_models", _scan_all_models)
    client = TestClient(
        create_app(StudioRuntimeSettings(max_sync_analysis_simulations=2)),
        base_url="http://127.0.0.1",
    )

    response = client.post("/api/models/scan/jobs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_mode"] == "async_job"
    assert payload["schema_version"] == "studio.model-scan.job.v1"
    assert isinstance(payload["job_id"], str)
    assert payload["job_id"].startswith("sj_")
    assert payload["status_route"] == f"/api/studio/jobs/{payload['job_id']}"
    assert payload["job"]["kind"] == "model_scan"
