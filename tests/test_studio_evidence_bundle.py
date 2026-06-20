# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence bundle tests

"""Tests for Studio evidence bundle exports."""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION,
    AuditEvent,
    JsonlAuditSink,
    StudioJobArtifact,
    StudioJobContext,
    StudioJobManager,
    StudioJobRecord,
    StudioRuntimeSettings,
    write_studio_evidence_bundle,
)
from sc_neurocore.studio.project import save_project

UTC = timezone.utc


def _simulation_payload() -> dict[str, object]:
    """Return a minimal Studio simulation response carrying run metadata."""

    return {
        "current_trace": [1.0, 1.0],
        "dt": 0.1,
        "n_steps": 2,
        "run_metadata": {
            "dt": 0.1,
            "evidence_classification": "simulation",
            "input_sha256": "1" * 64,
            "n_steps": 2,
            "result_sha256": "2" * 64,
            "sample_count": 2,
            "schema_version": "studio.simulation-run.v1",
            "source": "ode",
            "spike_count": 0,
            "state_variables": ["v"],
        },
        "spike_count": 0,
        "spikes": [],
        "states": {"v": [0.0, 0.1]},
        "time": [0.0, 0.1],
    }


def _analysis_payload() -> dict[str, object]:
    """Return a minimal Studio analysis response carrying analysis metadata."""

    return {
        "analysis_metadata": {
            "analysis_type": "fi_curve",
            "evidence_classification": "analysis",
            "input_sha256": "3" * 64,
            "output_keys": ["currents", "rates"],
            "result_sha256": "4" * 64,
            "schema_version": "studio.analysis-result.v1",
            "source": "ode",
        },
        "currents": [0.0, 1.0],
        "rates": [0.0, 10.0],
    }


def _client_with_evidence_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, TestClient]:
    """Create a Studio app with durable job, audit, and project roots."""

    projects_root = tmp_path / "projects"
    monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(projects_root))
    audit_path = tmp_path / "audit" / "studio.jsonl"
    settings = StudioRuntimeSettings(
        audit_log_path=str(audit_path),
        job_root_path=str(tmp_path / "jobs"),
        job_default_timeout_seconds=10.0,
    )
    app = create_app(settings)
    sink = JsonlAuditSink(audit_path)
    sink.record(
        AuditEvent(
            action="studio.test",
            route="/api/test",
            principal_id="operator",
            decision="allow",
            reason="authorized",
            request_id="req-test",
            timestamp_utc="2026-06-20T00:00:00Z",
        )
    )
    return app, TestClient(app, base_url="http://127.0.0.1")


def _job_manager(app: FastAPI) -> StudioJobManager:
    """Return the app-local Studio job manager."""

    return cast(StudioJobManager, app.state.studio_job_manager)


def _json_artifact(
    manager: StudioJobManager,
    job_id: str,
    relative_path: str,
) -> dict[str, object]:
    """Read one JSON job artifact through the verified artifact API."""

    payload = manager.read_artifact(job_id, relative_path)
    decoded = json.loads(payload.payload.decode("utf-8"))
    assert isinstance(decoded, dict)
    return cast(dict[str, object], decoded)


def test_write_studio_evidence_bundle_copies_project_job_audit_and_replay(
    tmp_path: Path,
) -> None:
    """Evidence bundle writer preserves project, job, audit, and artifact data."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=1.0,
    )

    def source_task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("compiler/result.json", '{"compiled": true}\n')
        return {"compiled": True}

    source_record = manager.submit(
        kind="compiler",
        owner="studio-compiler",
        request_id="req-1",
        task=source_task,
    )
    completed_source = manager.wait(source_record.job_id, timeout_seconds=2.0)
    bundle_context = StudioJobContext(
        job_id="sj_evidence",
        work_dir=tmp_path / "evidence",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )

    result = write_studio_evidence_bundle(
        bundle_context,
        project_payload={"name": "demo", "state": {"duration": 10}},
        simulation_payloads=(_simulation_payload(),),
        analysis_payloads=(_analysis_payload(),),
        job_records=(completed_source,),
        artifact_reader=manager.read_artifact,
        audit_export={"schema_version": "studio.audit.export.v1", "events": []},
        command_replay={"method": "POST", "path": "/api/compile"},
        clock=lambda: datetime(2026, 6, 20, tzinfo=UTC),
    )
    payload = result.to_public_dict()

    assert payload["schema_version"] == STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert payload["bundle_id"] == "seb_sj_evidence"
    assert "evidence/manifest.json" in result.artifact_paths
    assert "evidence/project.json" in result.artifact_paths
    assert "evidence/simulations/000.json" in result.artifact_paths
    assert "evidence/analyses/000.json" in result.artifact_paths
    assert f"evidence/jobs/{source_record.job_id}/record.json" in result.artifact_paths
    assert (
        f"evidence/jobs/{source_record.job_id}/artifacts/compiler/result.json"
        in result.artifact_paths
    )
    assert (tmp_path / "evidence" / "evidence" / "command-replay.json").is_file()
    assert "compiler/result.json" in json.dumps(payload)
    assert "simulation_result" in json.dumps(payload)
    assert "analysis_result" in json.dumps(payload)


def test_write_studio_evidence_bundle_rejects_invalid_json_and_artifact_state(
    tmp_path: Path,
) -> None:
    """Evidence bundle writer fails closed on unsafe replay and artifact inputs."""

    context = StudioJobContext(
        job_id="sj_evidence",
        work_dir=tmp_path / "evidence",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )
    with pytest.raises(ValueError, match="command replay"):
        write_studio_evidence_bundle(
            context,
            command_replay={"bad": math.nan},
        )
    with pytest.raises(ValueError, match="command replay"):
        write_studio_evidence_bundle(
            context,
            command_replay={"bad": object()},
        )
    with pytest.raises(ValueError, match="project payload"):
        write_studio_evidence_bundle(
            context,
            project_payload=cast(dict[str, object], {1: "bad"}),
        )
    with pytest.raises(ValueError, match="Studio simulation payload requires run metadata"):
        write_studio_evidence_bundle(
            context,
            simulation_payloads=({"time": []},),
        )
    invalid_simulation = _simulation_payload()
    invalid_simulation["run_metadata"] = {"schema_version": "legacy"}
    with pytest.raises(ValueError, match="unsupported run metadata"):
        write_studio_evidence_bundle(
            context,
            simulation_payloads=(invalid_simulation,),
        )
    with pytest.raises(ValueError, match="Studio analysis payload requires analysis metadata"):
        write_studio_evidence_bundle(
            context,
            analysis_payloads=({"rates": []},),
        )
    invalid_analysis = _analysis_payload()
    invalid_analysis["analysis_metadata"] = {"schema_version": "legacy"}
    with pytest.raises(ValueError, match="unsupported analysis metadata"):
        write_studio_evidence_bundle(
            context,
            analysis_payloads=(invalid_analysis,),
        )

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=1.0,
    )

    def source_task(job_context: StudioJobContext) -> dict[str, object]:
        job_context.write_artifact("compiler/result.json", "{}")
        return {}

    source_record = manager.submit(
        kind="compiler",
        owner="studio-compiler",
        request_id=None,
        task=source_task,
    )
    completed_source = manager.wait(source_record.job_id, timeout_seconds=2.0)
    with pytest.raises(ValueError, match="artifact reader"):
        write_studio_evidence_bundle(context, job_records=(completed_source,))
    unsafe_record = StudioJobRecord(
        job_id="sj_unsafe",
        kind="compiler",
        owner="operator",
        request_id=None,
        status="completed",
        created_at_utc="2026-06-20T00:00:00Z",
        artifacts=(
            StudioJobArtifact(
                relative_path="../escape.json",
                size_bytes=2,
                sha256="0" * 64,
            ),
        ),
    )
    with pytest.raises(ValueError, match="bundle-safe"):
        write_studio_evidence_bundle(
            context,
            job_records=(unsafe_record,),
            artifact_reader=manager.read_artifact,
        )


def test_studio_evidence_bundle_route_exports_selected_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admin evidence route creates a job-backed bundle with downloadable files."""

    app, client = _client_with_evidence_state(tmp_path, monkeypatch)
    save_project("demo", {"graph": {"populations": [], "projections": []}})
    compile_response = client.post(
        "/api/compile",
        json={
            "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
            "threshold": "v > -50",
            "reset": "v = -65",
            "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
        },
    )
    assert compile_response.status_code == 200
    simulation_response = client.post(
        "/api/simulate",
        json={
            "current": 1.0,
            "dt": 0.1,
            "duration": 1.0,
            "equations": ["dv/dt = I"],
            "init": {"v": 0.0},
        },
    )
    assert simulation_response.status_code == 200
    analysis_response = client.post(
        "/api/fi-curve",
        json={
            "duration": 1.0,
            "equations": ["dv/dt = I"],
            "i_max": 1.0,
            "i_min": 0.0,
            "i_steps": 2,
            "init": {"v": 0.0},
        },
    )
    assert analysis_response.status_code == 200
    source_records = [
        record
        for record in _job_manager(app).list_records()
        if record.owner == "studio-compiler"
    ]
    assert len(source_records) == 1

    response = client.post(
        "/api/studio/evidence/bundle",
        json={
            "project_name": "demo",
            "simulation_results": [simulation_response.json()],
            "analysis_results": [analysis_response.json()],
            "job_ids": [source_records[0].job_id],
            "include_audit": True,
            "audit_limit": 10,
            "command_replay": {
                "method": "POST",
                "path": "/api/compile",
                "request_body_sha256": "0" * 64,
            },
        },
    )
    body = response.json()
    manager = _job_manager(app)
    evidence_job_id = body["job_id"]
    manifest = _json_artifact(manager, evidence_job_id, "evidence/manifest.json")
    project_payload = _json_artifact(manager, evidence_job_id, "evidence/project.json")
    replay_payload = _json_artifact(manager, evidence_job_id, "evidence/command-replay.json")
    simulation_payload = _json_artifact(
        manager,
        evidence_job_id,
        "evidence/simulations/000.json",
    )
    analysis_payload = _json_artifact(
        manager,
        evidence_job_id,
        "evidence/analyses/000.json",
    )
    copied_result = _json_artifact(
        manager,
        evidence_job_id,
        f"evidence/jobs/{source_records[0].job_id}/artifacts/compiler/result.json",
    )
    encoded_body = json.dumps(body)

    assert response.status_code == 200
    assert body["schema_version"] == STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert body["bundle_id"] == f"seb_{evidence_job_id}"
    assert body["artifacts"]
    assert manifest["schema_version"] == STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert project_payload["name"] == "demo"
    assert simulation_payload["run_metadata"] == simulation_response.json()["run_metadata"]
    assert analysis_payload["analysis_metadata"] == analysis_response.json()["analysis_metadata"]
    assert replay_payload["path"] == "/api/compile"
    assert copied_result == compile_response.json()
    assert str(tmp_path) not in encoded_body
    assert "bearer_token" not in encoded_body
    assert "token_sha256" not in encoded_body


def test_studio_evidence_bundle_route_rejects_unknown_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evidence route reports missing project and job references clearly."""

    _app, client = _client_with_evidence_state(tmp_path, monkeypatch)

    missing_project = client.post(
        "/api/studio/evidence/bundle",
        json={"project_name": "missing", "include_audit": False},
    )
    missing_job = client.post(
        "/api/studio/evidence/bundle",
        json={"job_ids": ["sj_missing"], "include_audit": False},
    )

    assert missing_project.status_code == 404
    assert missing_job.status_code == 404
    assert missing_job.json()["detail"] == "job_not_found"


def test_studio_evidence_bundle_route_requires_configured_audit_export(
    tmp_path: Path,
) -> None:
    """Audit-inclusive evidence bundles require a persistent audit sink."""

    settings = StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs"))
    client = TestClient(create_app(settings), base_url="http://127.0.0.1")

    response = client.post("/api/studio/evidence/bundle", json={"include_audit": True})

    assert response.status_code == 409
    assert response.json()["detail"] == "audit_export_unavailable"
