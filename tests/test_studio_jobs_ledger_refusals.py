# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ledger custody refusals

"""Exercise retained lease, purge intent and reservation custody through the ledger API."""

import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_models import StudioJobRefused
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission

JOB = "sj_0000000000000001"


def _create(ledger: StudioJobLedger, *, job_id: str = JOB) -> StudioJobSubmission:
    return ledger.create(
        job_id=job_id,
        kind="analysis",
        actor="operator",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="thread",
    )


def test_a_live_peer_cannot_heartbeat_another_supervisors_lease(tmp_path: Path) -> None:
    """An actual child identity cannot claim a job created by the parent."""
    owner = StudioJobLedger(root=tmp_path / "jobs")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    peer = StudioJobLedger(root=owner.path.parent, supervisor=supervisor_identity(child.pid))
    try:
        original = _create(owner).record
        transitions = owner.transitions(JOB)
        with pytest.raises(StudioJobRefused) as refused:
            peer.heartbeat(JOB)
        assert str(refused.value) == f"Studio job {JOB} lease belongs to another supervisor."
        assert owner.record(JOB) == original
        assert owner.transitions(JOB) == transitions
        assert child.poll() is None
    finally:
        peer.close()
        owner.close()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10.0)


@pytest.mark.parametrize("custody", ["active", "prepared", "reserved"])
def test_ledger_purge_preserves_outstanding_custody(tmp_path: Path, custody: str) -> None:
    """A real retained intent or capacity reservation prevents destructive deletion."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    try:
        if custody == "reserved":
            admission.admit(
                job_id=JOB,
                kind="analysis",
                actor="operator",
                workspace="default",
                request_id=None,
                idempotency_key=None,
                experiment_sha256=None,
                admission=None,
                execution_model="thread",
            )
        else:
            _create(ledger)
        ledger.transition(JOB, "running")
        if custody != "active":
            ledger.transition(JOB, "completed", result={"answer": 42})
        if custody == "prepared":
            directory = ledger.path.parent / JOB
            directory.mkdir()
            identity = directory.stat()
            with ledger.transaction() as connection:
                connection.execute(
                    "INSERT INTO job_purges VALUES(?,?,?,?, 'prepared')",
                    (JOB, ledger.supervisor, identity.st_dev, identity.st_ino),
                )
        original = ledger.record(JOB)
        transitions = ledger.transitions(JOB)
        before = admission.snapshot()
        with pytest.raises(StudioJobRefused) as refused:
            ledger.delete(JOB)
        reason = {
            "active": f"Studio job {JOB} is not terminal and cannot be purged.",
            "prepared": f"Studio job {JOB} has a pending purge requiring recovery.",
            "reserved": f"Studio job {JOB} retains worker capacity and cannot be purged.",
        }[custody]
        assert str(refused.value) == reason
        assert ledger.record(JOB) == original
        assert ledger.transitions(JOB) == transitions
        assert admission.snapshot() == before
        if custody == "prepared":
            assert (ledger.path.parent / JOB).is_dir()
            assert (
                ledger.connection()
                .execute("SELECT state FROM job_purges WHERE job_id=?", (JOB,))
                .fetchone()[0]
                == "prepared"
            )
    finally:
        ledger.close()


def test_invalid_retained_job_identifier_cannot_discard_a_record(tmp_path: Path) -> None:
    """The manager refuses a retained identifier outside its generated ID grammar."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    try:
        _create(ledger, job_id="invalid-retained-id")
        ledger.transition("invalid-retained-id", "running")
        original = ledger.transition("invalid-retained-id", "completed")
        manager = StudioJobManager(
            root=ledger.path.parent,
            allowed_kinds=frozenset({"analysis"}),
            default_timeout_seconds=10.0,
        )
        try:
            with pytest.raises(StudioJobRefused) as refused:
                manager.purge_terminal_record(original.job_id)
            assert str(refused.value) == "Studio job path escapes the job root."
            assert manager.record(original.job_id) == original
            assert manager.status().pending_purge_count == 0
        finally:
            manager._ledger.close()
    finally:
        ledger.close()
