# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evidence seed transfer lifecycle

"""Real seed failure, source custody, admission reuse and API denial tests."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.api.evidence_jobs import (
    EvidenceSeedInputs,
    prepare_evidence_process_payload,
)
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioJobContext, StudioJobManager, StudioRuntimeSettings
from sc_neurocore.studio.platform.jobs_models import StudioJobArtifactPayload, StudioJobRecord

_TASK = "sc_neurocore.studio.api.evidence_jobs:execute_evidence_bundle_task"


@pytest.fixture
def source(tmp_path: Path) -> Iterator[tuple[StudioJobManager, StudioJobRecord]]:
    """Publish two actual source artifacts with only one admissible worker slot."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"evidence"}),
        default_timeout_seconds=15.0,
        max_concurrent_jobs=1,
        max_queued_jobs=0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("first.bin", b"first")
        context.write_artifact("second.bin", b"second")
        return {}

    job = manager.submit(kind="evidence", owner="studio-evidence", request_id=None, task=task)
    record = manager.wait(job.job_id, timeout_seconds=6.0)
    assert record.status == "completed"
    try:
        yield manager, record
    finally:
        for item in manager.list_records():
            if item.status in {"pending", "running", "cancelling"}:
                manager.cancel(item.job_id)
                manager.wait(item.job_id, timeout_seconds=16.0)
        manager._ledger.close()


def _payload(record: StudioJobRecord) -> dict[str, object]:
    """Prepare a complete export envelope using the production ingress owner."""
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
        "command_replay": None,
    }
    return prepare_evidence_process_payload(inputs, (record,), 1_000_000)


@pytest.mark.parametrize("fault", ["missing", "same_size_tamper", "truncated"])
def test_worker_seed_failure_preserves_source_and_reuses_only_slot(
    tmp_path: Path, source: tuple[StudioJobManager, StudioJobRecord], fault: str
) -> None:
    """Bad second seed refuses manifest; a subsequent real export still completes."""
    manager, record = source
    payload = _payload(record)
    original = EvidenceSeedInputs((record,), manager.read_artifact)
    seeds = dict(original)
    if fault == "missing":
        del seeds["bundle-0-1.bin"]
    elif fault == "same_size_tamper":
        seeds["bundle-0-1.bin"] = b"SECOND"
    else:
        seeds["bundle-0-1.bin"] = b"short"
    job = manager.submit_process_task(
        kind="evidence",
        owner="studio-evidence",
        request_id="fault",
        task_path=_TASK,
        payload=payload,
        seed_inputs=seeds,
    )
    failed = manager.wait(job.job_id, timeout_seconds=16.0)
    assert failed.status == "failed"
    assert failed.result is None and not failed.artifacts
    assert not (tmp_path / "jobs" / failed.job_id / "evidence/manifest.json").exists()
    assert manager.record(record.job_id) == record
    assert manager.read_artifact(record.job_id, "first.bin").payload == b"first"
    assert manager.read_artifact(record.job_id, "second.bin").payload == b"second"
    assert manager.status().admission["running"] == 0
    next_job = manager.submit_process_task(
        kind="evidence",
        owner="studio-evidence",
        request_id="next",
        task_path=_TASK,
        payload=payload,
        seed_inputs=original,
    )
    completed = manager.wait(next_job.job_id, timeout_seconds=16.0)
    assert completed.status == "completed"
    assert (
        manager.read_artifact(
            completed.job_id, f"evidence/jobs/{record.job_id}/artifacts/second.bin"
        ).payload
        == b"second"
    )
    assert manager.record(failed.job_id) == failed


def test_seed_mapping_reads_on_demand_without_caching_payloads(
    source: tuple[StudioJobManager, StudioJobRecord],
) -> None:
    """Metadata iteration does no reads; actual byte lookups revalidate each time."""
    manager, record = source
    calls: list[tuple[str, str]] = []

    def reader(job_id: str, path: str) -> StudioJobArtifactPayload:
        calls.append((job_id, path))
        return manager.read_artifact(job_id, path)

    seeds = EvidenceSeedInputs((record,), reader)
    assert len(seeds) == 2
    assert list(seeds) == ["bundle-0-0.bin", "bundle-0-1.bin"]
    assert calls == []
    assert seeds["bundle-0-1.bin"] == b"second"
    assert seeds["bundle-0-1.bin"] == b"second"
    assert calls == [(record.job_id, "second.bin"), (record.job_id, "second.bin")]
    with pytest.raises(KeyError):
        seeds["unlisted.bin"]
    assert len(calls) == 2


def test_evidence_policy_denial_creates_no_export_job(tmp_path: Path) -> None:
    """Existing development-mode ADMIN policy refuses before evidence admission."""
    app = create_app(
        StudioRuntimeSettings(
            job_root_path=str(tmp_path / "jobs"),
            enforce_route_policies=True,
            audit_log_path=str(tmp_path / "audit.jsonl"),
        )
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        manager = cast(StudioJobManager, app.state.studio_job_manager)
        response = client.post(
            "/api/studio/evidence/bundle",
            json={"include_audit": False},
            headers={"x-studio-principal": "viewer", "x-studio-roles": "studio.viewer"},
        )
        assert response.status_code == 403
        assert manager.list_records() == ()
        assert manager.status().admission["running"] == 0


def test_negative_source_size_refuses_before_seed_transfer(
    source: tuple[StudioJobManager, StudioJobRecord],
) -> None:
    """An invalid captured size cannot subtract from the aggregate input budget."""
    manager, record = source
    invalid = replace(record, artifacts=(replace(record.artifacts[0], size_bytes=-1),))
    with pytest.raises(ValueError, match="negative size"):
        _payload(invalid)
    assert manager.record(record.job_id) == record
    assert len(manager.list_records()) == 1


@pytest.mark.parametrize("fault", ["metadata", "length", "digest"])
def test_seed_mapping_rejects_reader_disagreement(
    source: tuple[StudioJobManager, StudioJobRecord], fault: str
) -> None:
    """Faulty reader declarations or bytes cannot overwrite captured source custody."""
    manager, record = source

    def reader(job_id: str, path: str) -> StudioJobArtifactPayload:
        actual = manager.read_artifact(job_id, path)
        if fault == "metadata":
            return replace(actual, artifact=replace(actual.artifact, relative_path="wrong.bin"))
        if fault == "length":
            return replace(actual, payload=b"x")
        return replace(actual, payload=actual.payload.upper())

    seeds = EvidenceSeedInputs((record,), reader)
    with pytest.raises(ValueError, match="source record"):
        seeds["bundle-0-0.bin"]
    assert manager.record(record.job_id) == record
    assert manager.read_artifact(record.job_id, "first.bin").payload == b"first"
