# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Worker registration identity validation

"""Refused registration preserves real job custody and permits a valid retry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.jobs_worker_custody import register_worker


@pytest.mark.parametrize("fault", ["malformed", "zero-token", "wrong-pid"])
def test_refused_worker_identity_preserves_custody_and_allows_retry(
    tmp_path: Path, fault: str
) -> None:
    """Validate a real session leader against an actual admitted process job."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    job_id = "sj_0123456789abcdef"
    admission.admit(
        job_id=job_id,
        kind="analysis",
        actor="owner",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
    )
    ledger.transition(job_id, "running")
    before = ledger.record(job_id)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(15)"], start_new_session=True
    )
    try:
        valid_identity = supervisor_identity(child.pid)
        host, pid, token = valid_identity.split(":", 2)
        identity = {
            "malformed": "incomplete",
            "zero-token": f"{host}:{pid}:0",
            "wrong-pid": f"{host}:{int(pid) + 1}:{token}",
        }[fault]
        with pytest.raises(ValueError, match="identity"):
            register_worker(ledger, job_id, ledger.supervisor, identity, child.pid)
        assert ledger.connection().execute("SELECT * FROM job_workers").fetchall() == []
        assert ledger.record(job_id) == before
        assert admission.snapshot().running == 1
        register_worker(ledger, job_id, ledger.supervisor, valid_identity, child.pid)
        rows = ledger.connection().execute("SELECT * FROM job_workers").fetchall()
        assert len(rows) == 1
        assert rows[0]["worker_identity"] == valid_identity
        assert rows[0]["group_id"] == child.pid
        assert rows[0]["supervisor"] == ledger.supervisor
        assert ledger.record(job_id) == before
        assert admission.snapshot().running == 1
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=3.0)
        ledger.close()
