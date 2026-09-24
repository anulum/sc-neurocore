# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio model-scan analysis budget route tests

"""Route-level tests for Studio model-scan budget enforcement."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
import time

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.api import catalogue as catalogue_routes
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.models import list_models
from sc_neurocore.studio.platform.jobs_manager import StudioJobManager


@pytest.fixture
def model_scan_client(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[TestClient]:
    """Own the real scan runtime through completion or failed-test cleanup."""
    app = create_app(
        StudioRuntimeSettings(
            max_sync_analysis_simulations=2,
            job_default_timeout_seconds=float(getattr(request, "param", 30.0)),
            job_root_path=str(tmp_path),
        )
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        try:
            yield client
        finally:
            manager = app.state.studio_job_manager
            for record in manager.list_records():
                if record.status not in {"completed", "failed", "timed_out", "cancelled"}:
                    manager.cancel(record.job_id)
                    manager.wait(record.job_id, 15.0)
            manager._ledger.close()


@pytest.mark.parametrize(
    ("model_scan_client", "stop"),
    [(30.0, "cancel"), (0.5, "timeout")],
    indirect=["model_scan_client"],
)
def test_http_scan_reports_stopped_job_and_reusable_capacity(
    model_scan_client: TestClient, tmp_path: Path, stop: str
) -> None:
    """HTTP-submitted scans reflect real peer cancellation or execution timeout.

    Cancellation uses the existing public manager because no HTTP cancellation
    endpoint exists. This verifies backend lifecycle, not a browser Stop action.
    """
    client = model_scan_client
    response = client.post("/api/models/scan/jobs", headers={"x-request-id": "scan-custody-trace"})
    assert response.status_code == 200
    receipt = response.json()
    assert receipt["job"]["request_id"] == "scan-custody-trace"
    assert receipt["job"]["owner"] == "studio"
    observer = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"model_scan"}), default_timeout_seconds=30.0
    )
    try:
        if stop == "cancel":
            observer.cancel(receipt["job_id"])
        terminal = observer.wait(receipt["job_id"], 15.0)
        assert terminal.status == ("cancelled" if stop == "cancel" else "timed_out")
        visible = client.get(receipt["status_route"])
        assert visible.status_code == 200
        assert visible.json()["status"] == terminal.status
        assert visible.json()["execution_model"] == "process"
        assert visible.json()["result"] is None
        assert visible.json()["artifacts"] == []
        deadline = time.monotonic() + 5.0
        while observer.status().admission["running"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert observer.status().admission["running"] == 0
        assert observer.status().admission["queued"] == 0
        health = client.get("/api/studio/jobs/status")
        assert health.status_code == 200
        assert health.json()["unreaped_workers"] == []
        history = observer.transitions(receipt["job_id"])
        assert observer.cancel(receipt["job_id"]) == terminal
        observer.reconcile()
        assert observer.transitions(receipt["job_id"]) == history
        next_job = observer.submit_process_task(
            kind="model_scan",
            owner="studio",
            request_id=None,
            task_path="sc_neurocore.studio.api.model_scan_jobs:execute_model_scan_process_task",
            payload={"current": 10.0, "duration": 1.0},
        )
        assert observer.wait(next_job.job_id, 30.0).status == "completed"
        assert client.get(receipt["status_route"]).json() == visible.json()
    finally:
        for record in observer.list_records():
            if record.status not in {"completed", "failed", "timed_out", "cancelled"}:
                observer.cancel(record.job_id)
                observer.wait(record.job_id, 15.0)
        observer._ledger.close()


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

    def _scan_all_models(
        *,
        current: float,
        duration: float,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
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

    def _scan_all_models(
        *,
        current: float,
        duration: float,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        # The synchronous route has no job to cancel, so it passes no
        # callback. The stub still accepts one: a stub that did not would pass
        # here while the job route it shares a name with raised TypeError.
        assert should_stop is None
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


def test_model_scan_job_route_polls_to_completed_with_model_scan_schema(
    model_scan_client: TestClient,
) -> None:
    """model_scan jobs must complete with studio.model-scan.v1 evidence class."""

    import time

    client = model_scan_client

    response = client.post("/api/models/scan/jobs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_mode"] == "async_job"
    assert payload["schema_version"] == "studio.model-scan.job.v1"
    assert isinstance(payload["job_id"], str)
    assert payload["job_id"].startswith("sj_")
    status_route = payload["status_route"]
    assert status_route == f"/api/studio/jobs/{payload['job_id']}"

    # Observe the configured execution deadline, not the former mock's latency.
    deadline = time.monotonic() + 31.0
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        job_response = client.get(status_route)
        assert job_response.status_code == 200
        last = job_response.json()
        if last.get("status") == "completed":
            break
        if last.get("status") in {"failed", "timed_out", "cancelled"}:
            pytest.fail(f"model_scan job failed: {last.get('status')}: {last.get('error')}")
        time.sleep(0.05)
    else:
        pytest.fail(f"model_scan job did not complete: {last}")

    assert last["status"] == "completed"
    assert last["kind"] == "model_scan"
    assert last["execution_model"] == "process"
    result = last["result"]
    assert isinstance(result, dict)
    assert result["schema_version"] == "studio.model-scan.v1"
    assert result["scan_metadata"]["evidence_classification"] == "analysis"
    assert result["scan_metadata"]["schema_version"] == "studio.model-scan.v1"
    expected = {entry["name"] for entry in list_models()}
    assert {entry["name"] for entry in result["models"]} == expected
    metadata = result["scan_metadata"]
    assert metadata["model_count"] == len(expected)
    assert metadata["error_count"] == sum(entry["pattern"] == "error" for entry in result["models"])
    assert metadata["current"] == 10.0
    assert metadata["duration"] == 100.0
