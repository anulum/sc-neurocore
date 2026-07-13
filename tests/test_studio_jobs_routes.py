# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job sandbox contract tests

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.jobs import (
    StudioJobContext,
    StudioJobManager,
)


def test_studio_job_status_endpoint_is_path_free(tmp_path: Path) -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            job_root_path=str(tmp_path / "jobs"),
            job_default_timeout_seconds=3.0,
        )
    )
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get("/api/studio/jobs/status")

    assert response.status_code == 200
    assert response.json() == {
        "active_count": 0,
        "allowed_kinds": ["compiler", "evidence", "synthesis", "training"],
        "completed_count": 0,
        "configured": True,
        "failed_count": 0,
        "process_count": 0,
        "resource_profiles": [
            {
                "default_timeout_seconds": 3.0,
                "execution_models": ["thread", "process"],
                "kind": "compiler",
                "max_artifact_bytes": 16777216,
            },
            {
                "default_timeout_seconds": 3.0,
                "execution_models": ["thread", "process"],
                "kind": "evidence",
                "max_artifact_bytes": 16777216,
            },
            {
                "default_timeout_seconds": 3.0,
                "execution_models": ["thread", "process"],
                "kind": "synthesis",
                "max_artifact_bytes": 16777216,
            },
            {
                "default_timeout_seconds": 3.0,
                "execution_models": ["thread", "process"],
                "kind": "training",
                "max_artifact_bytes": 16777216,
            },
        ],
        "schema_version": "studio.jobs.status.v1",
        "thread_count": 0,
        "timed_out_count": 0,
    }
    assert str(tmp_path) not in response.text


def test_studio_job_artifact_endpoint_is_admin_gated_and_integrity_checked(
    tmp_path: Path,
) -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
            job_default_timeout_seconds=3.0,
        )
    )
    manager = cast(StudioJobManager, app.state.studio_job_manager)

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"artifact body")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)
    client = TestClient(app, base_url="http://127.0.0.1")

    missing_principal = client.get(f"/api/studio/jobs/{record.job_id}/artifacts/reports/result.txt")
    allowed = client.get(
        f"/api/studio/jobs/{record.job_id}/artifacts/reports/result.txt",
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )

    assert missing_principal.status_code == 401
    assert missing_principal.json()["detail"] == "missing_principal"
    assert allowed.status_code == 200
    assert allowed.content == b"artifact body"
    assert allowed.headers["content-type"] == "application/octet-stream"
    assert allowed.headers["x-studio-artifact-size"] == str(len(b"artifact body"))
    assert (
        allowed.headers["x-studio-artifact-sha256"] == hashlib.sha256(b"artifact body").hexdigest()
    )
    assert str(tmp_path) not in str(allowed.headers)
    assert str(tmp_path) not in allowed.text


def test_studio_job_list_and_detail_endpoints_are_admin_gated_and_path_free(
    tmp_path: Path,
) -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
            job_default_timeout_seconds=3.0,
        )
    )
    manager = cast(StudioJobManager, app.state.studio_job_manager)

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"artifact body")
        return {"ok": True}

    record = manager.submit(
        kind="compiler",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)
    client = TestClient(app, base_url="http://127.0.0.1")
    admin_headers = {"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"}

    denied = client.get("/api/studio/jobs")
    listed = client.get("/api/studio/jobs", headers=admin_headers)
    detailed = client.get(f"/api/studio/jobs/{record.job_id}", headers=admin_headers)
    missing = client.get("/api/studio/jobs/sj_missing", headers=admin_headers)
    missing_artifact = client.get(
        "/api/studio/jobs/sj_missing/artifacts/reports/result.txt",
        headers=admin_headers,
    )

    assert denied.status_code == 401
    assert listed.status_code == 200
    assert listed.json()["schema_version"] == "studio.jobs.list.v1"
    assert listed.json()["jobs"][0]["job_id"] == record.job_id
    assert listed.json()["jobs"][0]["execution_model"] == "thread"
    assert listed.json()["jobs"][0]["artifacts"][0]["relative_path"] == "reports/result.txt"
    assert detailed.status_code == 200
    assert detailed.json()["job_id"] == record.job_id
    assert detailed.json()["execution_model"] == "thread"
    assert detailed.json()["status"] == "completed"
    assert missing.status_code == 404
    assert missing.json()["detail"] == "job_not_found"
    assert missing_artifact.status_code == 404
    assert missing_artifact.json()["detail"] == "job_artifact_not_found"
    assert str(tmp_path) not in listed.text
    assert str(tmp_path) not in detailed.text


def test_studio_job_artifact_endpoint_uses_generic_integrity_errors(tmp_path: Path) -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            enforce_route_policies=True,
            job_root_path=str(tmp_path / "jobs"),
            job_default_timeout_seconds=3.0,
        )
    )
    manager = cast(StudioJobManager, app.state.studio_job_manager)

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"original")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)
    (tmp_path / "jobs" / record.job_id / "reports" / "result.txt").write_bytes(b"tampered")
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.get(
        f"/api/studio/jobs/{record.job_id}/artifacts/reports/result.txt",
        headers={"x-studio-principal": "admin-1", "x-studio-roles": "studio.admin"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "job_artifact_unavailable"
    assert str(tmp_path) not in response.text
