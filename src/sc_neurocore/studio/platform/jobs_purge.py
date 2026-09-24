# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Recoverable job artifact purge

"""Stage artifacts before deleting records; restore them if deletion is refused."""

from __future__ import annotations

from sc_neurocore.studio.platform.jobs_ledger_writes import delete_job, require_purgeable
from sc_neurocore.studio.platform.jobs_ledger_schema import TERMINAL_STATUSES
from sc_neurocore.studio.platform.jobs_purge_recovery import PurgeCustody, _matches, recover_purges
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord, StudioJobRejected
from sc_neurocore.studio.platform import jobs_purge_paths


def _forget_purged_job(manager: PurgeCustody, job_id: str) -> None:
    """Discard local handles only after durable deletion is confirmed."""
    with manager._lock:
        manager._done_events.pop(job_id, None)
        manager._cancel_events.pop(job_id, None)
        manager._unreaped_workers.discard(job_id)


def purge_terminal_job(manager: PurgeCustody, job_id: str) -> StudioJobRecord:
    """Purge unreserved job custody without erasing files before a database refusal.

    A sibling staging directory retains the bytes until database deletion
    commits. Existing staging paths are never overwritten. If restoring after
    an error would overwrite a new path, retain the stage and report failure.
    """
    record = manager._ledger.record(job_id)
    if record.status not in TERMINAL_STATUSES:
        raise StudioJobRejected("Studio active jobs cannot be purged.")
    if (manager._root / record.job_id).is_symlink():
        raise StudioJobRejected("Studio job purge target cannot be a symlink.")
    try:
        work_dir = manager._job_work_dir(record.job_id)
    except ValueError as exc:
        raise StudioJobRejected(str(exc)) from exc
    staged = work_dir.with_name(f".purge-{work_dir.name}")
    prepared = False
    try:
        with manager._ledger.transaction() as connection:
            require_purgeable(connection, job_id)
            if staged.exists() or staged.is_symlink():
                raise StudioJobRejected("Studio job has a pending purge requiring recovery.")
            stat = work_dir.stat() if work_dir.exists() else None
            if work_dir.is_symlink() or (stat is not None and not work_dir.is_dir()):
                raise StudioJobRejected("Studio job purge target is not a directory.")
            connection.execute(
                "INSERT INTO job_purges VALUES(?,?,?,?, 'prepared')",
                (
                    job_id,
                    manager._ledger.supervisor,
                    None if stat is None else stat.st_dev,
                    None if stat is None else stat.st_ino,
                ),
            )
            prepared = True
        with manager._ledger.transaction() as connection:
            require_purgeable(connection, job_id, purge_supervisor=manager._ledger.supervisor)
            if staged.exists() or staged.is_symlink():
                raise StudioJobRejected("Studio job has a pending purge requiring recovery.")
            if work_dir.exists():
                if not work_dir.is_dir() or work_dir.is_symlink():
                    raise StudioJobRejected("Studio job purge target is not a directory.")
                current = work_dir.stat()
                if stat is None or (current.st_dev, current.st_ino) != (stat.st_dev, stat.st_ino):
                    raise StudioJobRejected("Studio job purge directory identity changed.")
                if not jobs_purge_paths.move_without_replace(work_dir, staged):
                    raise StudioJobRejected("Studio job has a pending purge requiring recovery.")
                if not _matches(staged, stat.st_dev, stat.st_ino):
                    raise StudioJobRejected(
                        "Studio job purge directory identity changed during move."
                    )
            elif stat is not None:
                raise StudioJobRejected("Studio job purge directory disappeared.")
            jobs_purge_paths.sync_directory(work_dir.parent)
            delete_job(manager._ledger, job_id, connection=connection)
            connection.execute("UPDATE job_purges SET state='committed' WHERE job_id=?", (job_id,))
    except BaseException:
        if not prepared:
            raise
        # The ledger transaction already rolled back whatever it began.
        connection = manager._ledger.connection()
        if connection.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone() is None:
            _forget_purged_job(manager, job_id)
        recover_purges(manager, own_job=job_id)
        raise
    _forget_purged_job(manager, job_id)
    if job_id not in recover_purges(manager, own_job=job_id):
        pending = (
            manager._ledger.connection()
            .execute("SELECT 1 FROM job_purges WHERE job_id=?", (job_id,))
            .fetchone()
        )
        if pending is not None:
            raise StudioJobRejected("Studio purge cleanup remains pending recovery.")
    return record
