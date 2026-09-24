# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job contracts under real inputs and damaged state

"""Contract edges of the job ledger and manager, driven by real inputs.

Damaged stored state is written for real through a second SQLite connection;
misuse of module-level writers passes real foreign or inactive connections.
"""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_admission import StudioJobAdmission
from sc_neurocore.studio.platform.jobs_admission_replay import StorageAdmissionReplay
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission
from sc_neurocore.studio.platform.jobs_ledger_writes import delete_job, transition_job
from sc_neurocore.studio.platform.jobs_ledger_creation import create_job
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifactUnavailable,
    StudioJobRejected,
)
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot


def _manager(root: Path) -> StudioJobManager:
    return StudioJobManager(
        root=root,
        allowed_kinds=frozenset({"analysis", "training"}),
        default_timeout_seconds=30.0,
    )


def _damage(path: Path, statement: str, *parameters: object) -> None:
    with closing(sqlite3.connect(path, isolation_level=None)) as other:
        other.execute("PRAGMA ignore_check_constraints=ON")
        other.execute(statement, parameters)


def test_storage_identity_names_the_opened_root_and_database(tmp_path: Path) -> None:
    """The recorded identity is the device and inode of the real root and file."""
    ledger = StudioJobLedger(root=tmp_path)
    try:
        root, database = tmp_path.stat(), ledger.path.stat()
        assert ledger.storage_identity == (
            root.st_dev,
            root.st_ino,
            database.st_dev,
            database.st_ino,
        )
    finally:
        ledger.close()


def test_writers_refuse_foreign_connections_and_empty_owners(tmp_path: Path) -> None:
    """Module-level writers act only inside this ledger's own transaction and owner."""
    ledger = StudioJobLedger(root=tmp_path)
    foreign = sqlite3.connect(":memory:")
    try:
        foreign.execute("BEGIN")
        with pytest.raises(ValueError, match="this ledger's active transaction"):
            delete_job(ledger, "sj_0000000000000001", connection=foreign)
        for connection in (foreign, ledger.connection()):
            with pytest.raises(ValueError, match="this ledger's active transaction"):
                transition_job(ledger, "sj_0000000000000001", "failed", connection=connection)
        with pytest.raises(ValueError, match="lease owner must be nonempty"):
            create_job(
                ledger,
                job_id="sj_0000000000000001",
                kind="analysis",
                actor="owner",
                workspace="default",
                request_id=None,
                idempotency_key=None,
                experiment_sha256=None,
                admission=None,
                execution_model="thread",
                lease_owner="",
            )
        assert ledger.list_records() == ()
    finally:
        foreign.rollback()
        foreign.close()
        ledger.close()


def test_releasing_an_idle_admission_only_wakes_waiters() -> None:
    """A release without a running job never makes the count negative."""
    admission = StudioJobAdmission(max_concurrent=1, max_queued=0)
    admission.release()
    assert admission.snapshot().running == 0


def test_damaged_artifact_size_is_unavailable(tmp_path: Path) -> None:
    """A negative stored artifact size is an integrity failure, not a read."""
    manager = _manager(tmp_path)

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("proof.txt", "evidence")
        return {}

    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
    assert manager.wait(job.job_id, 10.0).status == "completed"
    stored = json.loads(
        manager._ledger.connection()
        .execute("SELECT artifacts FROM jobs WHERE job_id=?", (job.job_id,))
        .fetchone()[0]
    )
    stored[0]["size_bytes"] = -1
    _damage(
        manager.ledger_path,
        "UPDATE jobs SET artifacts=? WHERE job_id=?",
        json.dumps(stored),
        job.job_id,
    )
    with pytest.raises(StudioJobArtifactUnavailable, match="integrity"):
        manager.read_artifact(job.job_id, "proof.txt")


def _admit_replay(admission: SharedJobAdmission, job_id: str) -> StudioJobSubmission:
    """Admit one process job under the same replay identity."""
    return admission.admit(
        job_id=job_id,
        kind="analysis",
        actor="studio-service",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
        replay=StorageAdmissionReplay("operator", "request", "a" * 64),
    )


@pytest.mark.parametrize("damage", ["record", "outcome"])
def test_damaged_admission_replay_is_refused(tmp_path: Path, damage: str) -> None:
    """A stored replay that no longer decodes is refused, never replayed."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    try:
        _admit_replay(admission, "sj_0000000000000001")
        statement = (
            "UPDATE storage_admission_replays SET record_json='not json'"
            if damage == "record"
            else "UPDATE storage_admission_replays SET record_json=NULL"
        )
        _damage(ledger.path, statement)
        with pytest.raises(StudioJobRejected, match="Stored admission replay is invalid"):
            _admit_replay(admission, "sj_0000000000000002")
    finally:
        ledger.close()


def _training_config() -> dict[str, object]:
    from sc_neurocore.studio.training_contract import resolve_training_config

    return resolve_training_config({}).to_public_dict()


def test_process_training_configuration_is_bound_to_training_jobs(tmp_path: Path) -> None:
    """A snapshot is admitted only for training jobs whose payload it describes."""
    manager = _manager(tmp_path)
    config = _training_config()
    with pytest.raises(StudioJobRejected, match="Only training jobs"):
        manager.submit_process_task(
            kind="analysis",
            owner="owner",
            request_id=None,
            task_path="tests.studio_job_tasks:process_echo_task",
            payload={"config": config},
            training_config=config,
        )
    job = manager.submit_process_task(
        kind="training",
        owner="owner",
        request_id=None,
        task_path="tests.studio_job_tasks:process_echo_task",
        payload={"config": config},
        training_config=config,
    )
    assert manager.wait(job.job_id, 30.0).status == "completed"
    assert manager.record(job.job_id).training_config == config


def _echo(manager: StudioJobManager, key: str) -> str:
    return manager.submit_process_task(
        kind="analysis",
        owner="owner",
        request_id=None,
        task_path="tests.studio_job_tasks:process_echo_task",
        payload={"answer": 42},
        idempotency_key=key,
    ).job_id


def test_duplicate_process_request_returns_the_first_job(tmp_path: Path) -> None:
    """A retried process request with the same idempotency key runs once."""
    manager = _manager(tmp_path)
    first = _echo(manager, "same-request")
    assert manager.wait(first, None).status == "completed"
    assert _echo(manager, "same-request") == first
    assert len(manager.list_records()) == 1


def test_oversized_training_snapshot_is_refused(tmp_path: Path) -> None:
    """A public snapshot cannot carry a training configuration beyond the bound."""
    manager = _manager(tmp_path)
    config = _training_config()
    job = manager.submit_process_task(
        kind="training",
        owner="owner",
        request_id=None,
        task_path="tests.studio_job_tasks:process_echo_task",
        payload={"config": config},
        training_config=config,
    )
    public = manager.wait(job.job_id, 30.0).to_public_dict()
    assert decode_job_snapshot(public).training_config == config
    public["training_config"] = {**config, "padding": "x" * 4096}
    with pytest.raises(ValueError, match="exceeds the 4096-byte limit"):
        decode_job_snapshot(public)
