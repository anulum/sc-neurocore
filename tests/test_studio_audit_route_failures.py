# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio audit route failure tests

"""Exercise audit and evidence failures through public Studio routes."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    AuditSinkError,
    JsonlAuditSink,
    StudioJobManager,
    StudioJobRecord,
    StudioJobStatus,
    StudioRuntimeSettings,
)


@pytest.fixture(scope="module")
def configured_app(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[FastAPI, TestClient, dict[str, Any], dict[str, Any]]:
    """Return a configured client with one valid quarantine archive."""
    root = tmp_path_factory.mktemp("studio-audit-routes")
    application = create_app(
        StudioRuntimeSettings(
            audit_log_path=str(root / "audit" / "studio.jsonl"),
            job_root_path=str(root / "jobs"),
        )
    )
    client = TestClient(application, base_url="http://127.0.0.1")
    response = client.post(
        "/api/studio/audit/quarantine/archive",
        json={"limit": 1},
    )
    assert response.status_code == 200
    manager = cast(StudioJobManager, application.state.studio_job_manager)
    job_id = response.json()["job_id"]
    archive = json.loads(
        manager.read_artifact(
            job_id,
            "evidence/audit-quarantine/archive.json",
        ).payload
    )
    manifest = json.loads(
        manager.read_artifact(
            job_id,
            "evidence/audit-quarantine/manifest.json",
        ).payload
    )
    assert isinstance(archive, dict)
    assert isinstance(manifest, dict)
    return application, client, archive, manifest


@pytest.mark.parametrize(
    ("method", "route", "payload"),
    [
        ("get", "/api/studio/audit/export", None),
        ("get", "/api/studio/audit/quarantine/export", None),
        ("post", "/api/studio/audit/quarantine/archive", {"limit": 1}),
        ("post", "/api/studio/evidence/bundle", {"include_audit": True}),
    ],
)
def test_audit_exports_require_jsonl_sink(
    method: str,
    route: str,
    payload: dict[str, Any] | None,
) -> None:
    client = TestClient(create_app(), base_url="http://127.0.0.1")

    response = client.get(route) if method == "get" else client.post(route, json=payload)

    assert response.status_code == 409
    assert response.json()["detail"] == "audit_export_unavailable"


@pytest.mark.parametrize(
    "route",
    [
        "/api/studio/audit/export",
        "/api/studio/audit/quarantine/export",
    ],
)
@pytest.mark.parametrize("limit", [0, 1001])
def test_audit_exports_reject_out_of_range_limit(
    configured_app: tuple[FastAPI, TestClient, dict[str, Any], dict[str, Any]],
    route: str,
    limit: int,
) -> None:
    _, client, _, _ = configured_app

    response = client.get(route, params={"limit": limit})

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("sink_method", "method", "route", "payload", "detail"),
    [
        (
            "export_recent",
            "get",
            "/api/studio/audit/export",
            None,
            "audit_export_failed",
        ),
        (
            "export_recent",
            "post",
            "/api/studio/evidence/bundle",
            {"include_audit": True},
            "audit_export_failed",
        ),
        (
            "export_quarantine",
            "get",
            "/api/studio/audit/quarantine/export",
            None,
            "audit_quarantine_export_failed",
        ),
        (
            "export_quarantine",
            "post",
            "/api/studio/audit/quarantine/archive",
            {"limit": 1},
            "audit_quarantine_export_failed",
        ),
    ],
)
def test_audit_sink_failures_are_path_free(
    configured_app: tuple[FastAPI, TestClient, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    sink_method: str,
    method: str,
    route: str,
    payload: dict[str, Any] | None,
    detail: str,
) -> None:
    _, client, _, _ = configured_app

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise AuditSinkError("private path")

    monkeypatch.setattr(JsonlAuditSink, sink_method, _raise)
    response = client.get(route) if method == "get" else client.post(route, json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == detail
    assert "private path" not in response.text


@pytest.mark.parametrize(
    ("status", "expected_status", "expected_detail"),
    [
        ("pending", 503, "studio_job_wait_exceeded"),
        ("timed_out", 504, "studio_job_timed_out"),
        ("failed", 500, "studio_job_failed"),
    ],
)
@pytest.mark.parametrize("route_kind", ["archive", "restore", "evidence"])
def test_audit_worker_terminal_states_map_to_stable_failures(
    configured_app: tuple[FastAPI, TestClient, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    status: StudioJobStatus,
    expected_status: int,
    expected_detail: str,
    route_kind: str,
) -> None:
    _, client, archive, manifest = configured_app

    def _submit(self: StudioJobManager, **_kwargs: object) -> StudioJobRecord:
        return StudioJobRecord(
            job_id="audit-route-error",
            kind="evidence",
            owner="studio-audit",
            request_id=None,
            status="pending",
            execution_model="thread",
            created_at_utc="2026-07-11T00:00:00Z",
        )

    def _wait(
        self: StudioJobManager,
        job_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> StudioJobRecord:
        del self, timeout_seconds
        return StudioJobRecord(
            job_id=job_id,
            kind="evidence",
            owner="studio-audit",
            request_id=None,
            status=status,
            execution_model="thread",
            created_at_utc="2026-07-11T00:00:00Z",
        )

    monkeypatch.setattr(StudioJobManager, "submit", _submit)
    monkeypatch.setattr(StudioJobManager, "wait", _wait)
    if route_kind == "archive":
        response = client.post(
            "/api/studio/audit/quarantine/archive",
            json={"limit": 1},
        )
    elif route_kind == "restore":
        response = client.post(
            "/api/studio/audit/quarantine/archive/restore",
            json={"archive": archive, "manifest": manifest},
        )
    else:
        response = client.post(
            "/api/studio/evidence/bundle",
            json={"include_audit": False},
        )

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail
