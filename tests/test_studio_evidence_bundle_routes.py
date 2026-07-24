# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence bundle HTTP routes

"""Admin route contracts for evidence-bundle export and configuration gates."""

from __future__ import annotations

from tests.studio_evidence_bundle_support import *  # noqa: F403


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
    default_flow_run_response = client.post(
        "/api/presets/fpga_precision/default-flow/run",
        json={"action_overrides": {"auto_tune_adaptive_precision": {"target_error_percent": 0.05}}},
    )
    assert default_flow_run_response.status_code == 200
    default_flow_attestation_response = client.post(
        "/api/presets/fpga_precision/default-flow/attest",
        json={"run_result": default_flow_run_response.json()},
    )
    assert default_flow_attestation_response.status_code == 200
    source_records = [
        record for record in _job_manager(app).list_records() if record.owner == "studio-compiler"
    ]
    assert len(source_records) == 1

    response = client.post(
        "/api/studio/evidence/bundle",
        json={
            "project_name": "demo",
            "simulation_results": [simulation_response.json()],
            "analysis_results": [analysis_response.json()],
            "default_flow_runs": [default_flow_run_response.json()],
            "default_flow_attestations": [default_flow_attestation_response.json()],
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
    default_flow_run_payload = _json_artifact(
        manager,
        evidence_job_id,
        "evidence/default-flows/runs/000.json",
    )
    default_flow_attestation_payload = _json_artifact(
        manager,
        evidence_job_id,
        "evidence/default-flows/attestations/000.json",
    )
    copied_result = _json_artifact(
        manager,
        evidence_job_id,
        f"evidence/jobs/{source_records[0].job_id}/artifacts/compiler/result.json",
    )
    copied_action_evidence = _json_artifact(
        manager,
        evidence_job_id,
        f"evidence/jobs/{source_records[0].job_id}/artifacts/compiler/evidence.json",
    )
    encoded_body = json.dumps(body)

    assert response.status_code == 200
    assert body["schema_version"] == STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert body["bundle_id"] == f"seb_{evidence_job_id}"
    assert body["artifacts"]
    assert manifest["schema_version"] == STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert manifest["summary"] == body["summary"]
    assert body["summary"]["artifact_path_count"] == len(body["artifact_paths"])
    assert body["summary"]["entry_type_counts"]["simulation_result"] == 1
    assert body["summary"]["entry_type_counts"]["analysis_result"] == 1
    assert body["summary"]["entry_type_counts"]["default_flow_run"] == 1
    assert body["summary"]["evidence_classification_counts"]["analysis"] == 1
    assert body["summary"]["evidence_classification_counts"]["compile"] == 1
    assert body["summary"]["evidence_classification_counts"]["project_workspace"] == 1
    assert body["summary"]["evidence_classification_counts"]["simulation"] == 1
    assert body["summary"]["source_job_kind_counts"]["compiler"] == 1
    manifest_entries = cast(list[dict[str, object]], manifest["entries"])
    simulation_entry = next(
        entry for entry in manifest_entries if entry["type"] == "simulation_result"
    )
    analysis_entry = next(entry for entry in manifest_entries if entry["type"] == "analysis_result")
    project_entry = next(entry for entry in manifest_entries if entry["type"] == "project")
    assert simulation_entry["evidence_classification"] == "simulation"
    assert analysis_entry["evidence_classification"] == "analysis"
    assert project_entry["evidence_classification"] == "project_workspace"
    assert project_payload["name"] == "demo"
    assert simulation_payload["run_metadata"] == simulation_response.json()["run_metadata"]
    assert analysis_payload["analysis_metadata"] == analysis_response.json()["analysis_metadata"]
    assert default_flow_run_payload["schema_version"] == "sc-neurocore.studio.default-flow-run.v1"
    assert default_flow_run_payload["evidence_classification"] == "default_flow"
    assert default_flow_run_payload["status"] == "completed"
    assert (
        default_flow_attestation_payload["schema_version"]
        == "sc-neurocore.studio.default-flow-attestation.v1"
    )
    assert default_flow_attestation_payload["evidence_classification"] == "default_flow"
    assert default_flow_attestation_payload["status"] == "completed"
    assert copied_action_evidence["schema_version"] == "studio.action-evidence.v1"
    assert copied_action_evidence["evidence_classification"] == "compile"
    assert "action_evidence" in json.dumps(manifest)
    assert "default_flow_run" in json.dumps(manifest)
    assert "default_flow_attestation" in json.dumps(manifest)
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
