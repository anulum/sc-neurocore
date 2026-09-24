# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Proven-stopped capacity recovery

"""Reclaim reservations only when stored identity and process evidence suffice."""

from __future__ import annotations

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import TERMINAL_STATUSES
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_is_alive
from sc_neurocore.studio.platform.jobs_worker_recovery import worker_group_stopped


def recover_stopped_reservations(ledger: StudioJobLedger) -> tuple[str, ...]:
    """Release dead-owner queues and terminal in-process work after reconciliation.

    A dead supervisor proves its threads ended, not its child processes. Keep
    process reservations unless stored identity proves their group stopped.
    Unprobeable identities and expiry alone never justify release.
    The caller reconciles job state first; this function never invents outcomes.
    """
    released: list[str] = []
    with ledger.transaction() as connection:
        rows = connection.execute(
            "SELECT r.job_id,r.supervisor,r.state,j.job_id AS retained_job,"
            "j.status,j.execution_model,w.supervisor AS worker_supervisor,"
            "w.worker_identity,w.boot_id,w.group_id FROM admission_reservations r "
            "LEFT JOIN jobs j ON j.job_id=r.job_id LEFT JOIN job_workers w ON w.job_id=r.job_id"
        ).fetchall()
        for row in rows:
            owner = row["supervisor"]
            if owner is None or supervisor_is_alive(str(owner)) is not False:
                continue
            never_started = row["state"] == "queued" and row["retained_job"] is None
            stopped_thread = (
                row["execution_model"] == "thread" and row["status"] in TERMINAL_STATUSES
            )
            stopped_process = (
                row["execution_model"] == "process"
                and row["status"] in TERMINAL_STATUSES
                and row["worker_supervisor"] == owner
                and row["worker_identity"] is not None
                and worker_group_stopped(row["worker_identity"], row["boot_id"], row["group_id"])
            )
            if never_started or stopped_thread or stopped_process:
                connection.execute(
                    "DELETE FROM admission_reservations WHERE job_id=?", (row["job_id"],)
                )
                released.append(str(row["job_id"]))
    return tuple(released)
