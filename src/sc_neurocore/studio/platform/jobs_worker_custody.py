# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Durable isolated worker identity

"""Trusted authority commits observed worker identity before permitting task import."""

from __future__ import annotations

import os
from pathlib import Path

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import (
    supervisor_is_alive,
)


def register_worker(
    ledger: StudioJobLedger,
    job_id: str,
    expected_supervisor: str,
    worker_identity: str,
    group_id: int,
) -> None:
    """Bind an observed child to its admitted job on the trusted authority side.

    Persist host/PID/start-token identity, boot identity and process group in
    the job ledger. Refuse inactive jobs, mismatched ownership, absent capacity,
    duplicate workers and a supervisor whose liveness cannot be established.
    Registration is evidence of startup, not proof of later group termination.
    """
    if group_id <= 0 or os.getpgid(group_id) != group_id:
        raise ValueError("Managed worker must lead its own process group.")
    identity_parts = worker_identity.split(":", 2)
    if (
        len(identity_parts) != 3
        or identity_parts[1] != str(group_id)
        or worker_identity.endswith(":0")
    ):
        raise ValueError("Worker identity is not verified.")
    # The kernel always provides a boot identity on Linux.
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    with ledger.transaction() as connection:
        row = connection.execute(
            "SELECT j.status,j.execution_model,j.lease_owner,r.supervisor,r.state "
            "FROM jobs j JOIN admission_reservations r ON r.job_id=j.job_id "
            "WHERE j.job_id=?",
            (job_id,),
        ).fetchone()
        if (
            row is None
            or row["status"] != "running"
            or row["execution_model"] != "process"
            or row["lease_owner"] != expected_supervisor
            or row["supervisor"] != expected_supervisor
            or row["state"] != "running"
            or supervisor_is_alive(expected_supervisor) is not True
            or supervisor_is_alive(worker_identity) is not True
        ):
            raise ValueError("Worker has no live admitted supervisor.")
        connection.execute(
            "INSERT INTO job_workers VALUES(?,?,?,?,?)",
            (job_id, expected_supervisor, worker_identity, boot_id, group_id),
        )
