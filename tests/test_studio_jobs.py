# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job sandbox contract tests

from __future__ import annotations

import hashlib
import time
import threading
from pathlib import Path
from typing import cast

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.jobs import (
    StudioJobArtifactUnavailable,
    StudioJobContext,
    StudioJobManager,
    StudioJobRejected,
)


def test_studio_job_manager_completes_job_with_path_free_artifact_manifest(
    tmp_path: Path,
) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"ok")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)

    assert completed.status == "completed"
    assert completed.result == {"ok": True}
    assert len(completed.artifacts) == 1
    assert completed.artifacts[0].relative_path == "reports/result.txt"
    assert manager.list_snapshot().to_public_dict()["schema_version"] == "studio.jobs.list.v1"
    assert str(tmp_path) not in str(completed.to_public_dict())
    assert (tmp_path / "jobs" / record.job_id / "reports" / "result.txt").read_bytes() == b"ok"


def test_studio_job_manager_reads_manifest_declared_artifacts(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", "artifact body")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)

    payload = manager.read_artifact(record.job_id, "reports/result.txt")

    assert payload.payload == b"artifact body"
    assert payload.artifact.relative_path == "reports/result.txt"
    assert payload.artifact.sha256 == hashlib.sha256(b"artifact body").hexdigest()


def test_studio_job_manager_rejects_tampered_manifest_artifact(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

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

    with pytest.raises(StudioJobArtifactUnavailable, match="integrity"):
        manager.read_artifact(record.job_id, "reports/result.txt")


def test_studio_job_manager_rejects_artifact_path_traversal(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("../escape.txt", b"bad")
        return {"unreachable": True}

    record = manager.submit(
        kind="compiler",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)

    assert completed.status == "failed"
    assert completed.error == "Studio job artifact path escapes the job directory."
    assert not (tmp_path / "escape.txt").exists()


def test_studio_job_context_rejects_symlink_artifact_escape(tmp_path: Path) -> None:
    work_dir = tmp_path / "job"
    outside_dir = tmp_path / "outside"
    work_dir.mkdir()
    outside_dir.mkdir()
    (work_dir / "linked").symlink_to(outside_dir, target_is_directory=True)
    context = StudioJobContext(
        job_id="sj_test",
        work_dir=work_dir,
        cancel_event=threading.Event(),
    )

    context.check_cancelled()
    with pytest.raises(ValueError, match="escapes"):
        context.write_artifact("linked/escape.txt", b"bad")


def test_studio_job_manager_rejects_disallowed_job_kind(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    with pytest.raises(StudioJobRejected, match="not allowed"):
        manager.submit(
            kind="training",
            owner="operator-1",
            request_id="req-1",
            task=lambda _context: {},
        )


def test_studio_job_manager_rejects_invalid_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allowed job kind"):
        StudioJobManager(
            root=tmp_path / "jobs",
            allowed_kinds=frozenset(),
            default_timeout_seconds=1.0,
        )
    with pytest.raises(ValueError, match="timeout"):
        StudioJobManager(
            root=tmp_path / "jobs",
            allowed_kinds=frozenset({"synthesis"}),
            default_timeout_seconds=0,
        )


def test_studio_job_manager_rejects_non_positive_submit_timeout(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    with pytest.raises(StudioJobRejected, match="timeout"):
        manager.submit(
            kind="synthesis",
            owner="operator-1",
            request_id="req-1",
            task=lambda _context: {},
            timeout_seconds=0,
        )


def test_studio_job_manager_cancel_finished_job_is_noop(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )
    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=lambda _context: {"done": True},
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)

    cancelled = manager.cancel(record.job_id)

    assert completed.status == "completed"
    assert cancelled.status == "completed"
    assert cancelled.result == {"done": True}


def test_studio_job_manager_cancels_cooperative_job(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=2.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        while not context.cancelled:
            time.sleep(0.01)
        context.check_cancelled()
        return {"unreachable": True}

    record = manager.submit(
        kind="training",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )

    assert manager.cancel(record.job_id).status == "cancelling"
    completed = manager.wait(record.job_id, timeout_seconds=2.0)
    assert completed.status == "cancelled"


def test_studio_job_manager_times_out_cooperative_job(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=0.05,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        while not context.cancelled:
            time.sleep(0.01)
        context.check_cancelled()
        return {"unreachable": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)

    assert completed.status == "timed_out"
    assert completed.error == "Studio job exceeded its timeout."


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
        "allowed_kinds": ["compiler", "synthesis", "training"],
        "completed_count": 0,
        "configured": True,
        "failed_count": 0,
        "schema_version": "studio.jobs.status.v1",
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

    assert denied.status_code == 401
    assert listed.status_code == 200
    assert listed.json()["schema_version"] == "studio.jobs.list.v1"
    assert listed.json()["jobs"][0]["job_id"] == record.job_id
    assert listed.json()["jobs"][0]["artifacts"][0]["relative_path"] == "reports/result.txt"
    assert detailed.status_code == 200
    assert detailed.json()["job_id"] == record.job_id
    assert detailed.json()["status"] == "completed"
    assert missing.status_code == 404
    assert missing.json()["detail"] == "job_not_found"
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
