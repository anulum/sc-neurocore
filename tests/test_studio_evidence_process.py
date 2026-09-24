# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evidence process ingress contracts

"""Exercise complete source custody and explicit input refusal through HTTP."""

import json
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.api.evidence_jobs import (
    EvidenceInputLimitExceeded,
    prepare_evidence_process_payload,
)
from sc_neurocore.studio.platform.jobs_models import StudioJobArtifact, StudioJobRecord
from sc_neurocore.studio.platform import StudioJobContext, StudioJobManager, StudioRuntimeSettings
from sc_neurocore.studio.platform.settings import build_default_studio_runtime_settings


@pytest.mark.parametrize("budget", [256 * 1024 * 1024, 32])
def test_bundle_process_copies_complete_source_or_refuses_before_admission(
    tmp_path: Path, budget: int
) -> None:
    """API export preserves record/bytes/identity or refuses a small aggregate cap."""
    app = create_app(
        StudioRuntimeSettings(
            job_root_path=str(tmp_path / "jobs"),
            evidence_max_input_bytes=budget,
            job_default_timeout_seconds=15.0,
        )
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        manager = cast(StudioJobManager, app.state.studio_job_manager)

        def source(context: StudioJobContext) -> dict[str, object]:
            context.write_artifact("first.bin", b"first source bytes")
            context.write_artifact("nested/second.bin", b"second source bytes")
            return {"nested": {"retained": [1, 2.0, True]}}

        submitted = manager.submit(
            kind="evidence",
            owner="studio-evidence",
            request_id="source-request",
            workspace="source-workspace",
            idempotency_key="source-key",
            experiment_sha256="a" * 64,
            admission={"policy": "retained"},
            task=source,
        )
        record = manager.wait(submitted.job_id, timeout_seconds=6.0)
        assert record.status == "completed"
        response = client.post(
            "/api/studio/evidence/bundle",
            json={
                "job_ids": [record.job_id],
                "include_audit": False,
            },
        )
        if budget == 32:
            assert response.status_code == 413, response.text
            assert response.json()["detail"] == "studio_evidence_input_limit_exceeded"
            assert [r.job_id for r in manager.list_records()] == [record.job_id]
            return
        assert response.status_code == 200, response.text
        result = response.json()
        exported = manager.record(result["job_id"])
        assert exported.execution_model == "process"
        assert exported.owner == "studio-evidence"
        assert exported.request_id == response.headers["x-request-id"]
        base = f"evidence/jobs/{record.job_id}"
        snapshot = manager.read_artifact(exported.job_id, f"{base}/record.json")
        assert json.loads(snapshot.payload) == record.to_public_dict()
        for artifact in record.artifacts:
            copied = manager.read_artifact(
                exported.job_id, f"{base}/artifacts/{artifact.relative_path}"
            )
            original = manager.read_artifact(record.job_id, artifact.relative_path)
            assert copied.payload == original.payload
            assert copied.artifact.sha256 == artifact.sha256


@pytest.mark.parametrize("value", ["0", "-1", "", "1.5", "true"])
def test_evidence_budget_environment_refuses_invalid_values(value: str) -> None:
    """Invalid explicit values never silently become a default budget."""
    with pytest.raises(ValueError):
        build_default_studio_runtime_settings(
            {"SC_NEUROCORE_STUDIO_EVIDENCE_MAX_INPUT_BYTES": value}
        )


def test_evidence_budget_environment_configures_exact_bytes() -> None:
    """An operator byte override reaches validated runtime settings unchanged."""
    settings = build_default_studio_runtime_settings(
        {"SC_NEUROCORE_STUDIO_EVIDENCE_MAX_INPUT_BYTES": "12345"}
    )
    assert settings.evidence_max_input_bytes == 12345


@pytest.mark.parametrize("payload", [{}, {"inputs": {}, "records": [], "max_input_bytes": 1000}])
def test_evidence_worker_rejects_incomplete_envelope(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    """Direct malformed worker inputs fail without manufacturing bundle evidence."""
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs")))
    with TestClient(app, base_url="http://127.0.0.1"):
        manager = cast(StudioJobManager, app.state.studio_job_manager)
        submitted = manager.submit_process_task(
            kind="evidence",
            owner="studio-evidence",
            request_id=None,
            task_path="sc_neurocore.studio.api.evidence_jobs:execute_evidence_bundle_task",
            payload=payload,
            timeout_seconds=15.0,
        )
        record = manager.wait(submitted.job_id, timeout_seconds=16.0)
        assert record.status == "failed"
        assert record.result is None
        assert not record.artifacts
        assert not (tmp_path / "jobs" / record.job_id / "evidence").exists()


def test_bundle_refuses_tampered_source_and_retains_failed_job(tmp_path: Path) -> None:
    """Corrupt source bytes fail verified ingress with stable HTTP failure mapping."""
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs")))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        manager = cast(StudioJobManager, app.state.studio_job_manager)

        def source(context: StudioJobContext) -> dict[str, object]:
            context.write_artifact("source.bin", b"original")
            return {}

        submitted = manager.submit(
            kind="evidence", owner="studio-evidence", request_id=None, task=source
        )
        assert manager.wait(submitted.job_id, timeout_seconds=6.0).status == "completed"
        (tmp_path / "jobs" / submitted.job_id / "source.bin").write_bytes(b"tampered")
        response = client.post(
            "/api/studio/evidence/bundle",
            json={
                "job_ids": [submitted.job_id],
                "include_audit": False,
            },
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "studio_job_failed"
        failures = [r for r in manager.list_records() if r.job_id != submitted.job_id]
        assert len(failures) == 1
        assert failures[0].status == "failed"
        assert failures[0].result is None
        assert not failures[0].artifacts
        assert manager.status().admission["running"] == 0
        retry = client.post("/api/studio/evidence/bundle", json={"include_audit": False})
        assert retry.status_code == 200, retry.text
        assert manager.record(retry.json()["job_id"]).status == "completed"
        assert manager.record(failures[0].job_id) == failures[0]


@pytest.mark.parametrize("copies", [0, 1, 2])
def test_evidence_input_budget_counts_exact_metadata_and_every_seed_copy(copies: int) -> None:
    """Exact serialized bytes fit; one byte less refuses without dropping copies."""
    inputs: dict[str, object] = {
        "project_payload": None,
        "simulation_payloads": [],
        "analysis_payloads": [],
        "model_scan_payloads": [],
        "weight_restore_payloads": [],
        "weight_restore_attach_payloads": [],
        "default_flow_runs": [],
        "default_flow_attestations": [],
        "audit_export": None,
        "command_replay": {"note": "váhy"},
    }
    record = StudioJobRecord(
        job_id="sj_0123456789abcdef",
        kind="evidence",
        owner="studio-evidence",
        request_id=None,
        status="completed",
        execution_model="process",
        created_at_utc="2026-09-12T00:00:00Z",
        artifacts=(StudioJobArtifact("input.bin", 512, "a" * 64),),
    )
    records = (record,) * copies
    limit = 10000
    # Independent serialization oracle includes the digit width of the ceiling
    # itself and repeated selections. This fixture converges within three passes.
    for _ in range(5):
        envelope = {
            "inputs": inputs,
            "records": [r.to_public_dict() for r in records],
            "max_input_bytes": limit,
        }
        expected = (
            len(json.dumps(envelope, sort_keys=True, allow_nan=False).encode()) + 512 * copies
        )
        if limit == expected:
            break
        limit = expected
    assert limit == expected
    accepted = prepare_evidence_process_payload(inputs, records, limit)
    assert accepted == envelope
    with pytest.raises(EvidenceInputLimitExceeded):
        prepare_evidence_process_payload(inputs, records, limit - 1)
