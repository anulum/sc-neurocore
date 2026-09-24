# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Journal-owned purge recovery

"""Restore interrupted purges or finish committed cleanup using exact directory custody."""

from __future__ import annotations

import os
import stat
import threading
from _thread import LockType
from typing import Literal, Protocol
from pathlib import Path

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_is_alive
from sc_neurocore.studio.platform import jobs_purge_paths


class PurgeCustody(Protocol):
    """What purging and its recovery use from their owner.

    The ledger and custody root, and the owner's local handles of jobs it
    supervises, which a purge forgets. The embedded manager is one owner;
    the storage authority is another, with no local handles.
    """

    _ledger: StudioJobLedger
    _root: Path
    _lock: LockType
    _done_events: dict[str, threading.Event]
    _cancel_events: dict[str, threading.Event]
    _unreaped_workers: set[str]

    def _job_work_dir(self, job_id: str) -> Path:
        """Return the custody directory of one job under the root."""


def _matches(path: Path, device: int | None, inode: int | None) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    stat = path.stat()
    return (stat.st_dev, stat.st_ino) == (device, inode)


def _clear_directory(descriptor: int) -> None:
    """Clear contents relative to an already verified open directory.

    Subdirectories are opened relative to their parent without following a
    link and cleared the same way before they are removed, which is what
    ``shutil.rmtree(dir_fd=...)`` does from Python 3.11; doing it here keeps one
    path on every supported Python.
    """
    for name in os.listdir(descriptor):
        entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(entry.st_mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                _clear_directory(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


def _remove_owned_directory(path: Path, device: int | None, inode: int | None) -> bool:
    """Never traverse replacement root contents after the pathname identity check."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except (FileNotFoundError, NotADirectoryError):
        return False
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (device, inode):
            return False
        _clear_directory(descriptor)
        if not _matches(path, device, inode):
            return False
        path.rmdir()
        # A pathname can be exchanged after _matches. Success must describe
        # the verified open object, not removal of an unrelated replacement.
        return os.fstat(descriptor).st_nlink == 0
    finally:
        os.close(descriptor)


def _recover_phase(
    manager: PurgeCustody, job_id: str, own_job: str | None
) -> Literal["pending", "advance", "resolved"]:
    """Commit one recovery phase with fresh custody checks under the writer lock."""
    with manager._ledger.transaction() as connection:
        row = connection.execute("SELECT * FROM job_purges WHERE job_id=?", (job_id,)).fetchone()
        if row is None or row["state"] == "ambiguous":
            return "pending"
        owned = row["supervisor"] == manager._ledger.supervisor and (
            own_job == job_id or row["state"] != "prepared"
        )
        if not owned and supervisor_is_alive(row["supervisor"]) is not False:
            return "pending"
        work_dir = manager._job_work_dir(job_id)
        staged = work_dir.with_name(f".purge-{work_dir.name}")
        present = connection.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        stage_exists = staged.exists() or staged.is_symlink()
        original_exists = work_dir.exists() or work_dir.is_symlink()
        if row["state"] == "prepared" and present is not None:
            if stage_exists:
                if original_exists or not _matches(staged, row["device"], row["inode"]):
                    return "pending"
                if not jobs_purge_paths.move_without_replace(staged, work_dir):
                    return "pending"
                if not _matches(work_dir, row["device"], row["inode"]):
                    return "pending"
            elif original_exists:
                if not _matches(work_dir, row["device"], row["inode"]):
                    return "pending"
            elif row["inode"] is not None or row["device"] is not None:
                return "pending"
            jobs_purge_paths.sync_directory(work_dir.parent)
            connection.execute("DELETE FROM job_purges WHERE job_id=?", (job_id,))
            return "resolved"
        if present is not None or row["state"] == "prepared":
            return "pending"
        has_identity = row["inode"] is not None or row["device"] is not None
        conflict = original_exists
        if row["state"] == "removed":
            conflict = conflict or stage_exists
        else:
            conflict = conflict or (
                not _matches(staged, row["device"], row["inode"]) if stage_exists else has_identity
            )
        if conflict:
            connection.execute("UPDATE job_purges SET state='ambiguous' WHERE job_id=?", (job_id,))
            return "pending"
        if row["state"] == "committed":
            connection.execute(
                "UPDATE job_purges SET state='cleanup_started' WHERE job_id=?", (job_id,)
            )
            return "advance"
        if row["state"] == "cleanup_started":
            if stage_exists and not _remove_owned_directory(staged, row["device"], row["inode"]):
                connection.execute(
                    "UPDATE job_purges SET state='ambiguous' WHERE job_id=?", (job_id,)
                )
                return "pending"
            jobs_purge_paths.sync_directory(work_dir.parent)
            connection.execute("UPDATE job_purges SET state='removed' WHERE job_id=?", (job_id,))
            return "advance"
        connection.execute("DELETE FROM job_purges WHERE job_id=?", (job_id,))
        return "resolved"


def recover_purges(manager: PurgeCustody, *, own_job: str | None = None) -> tuple[str, ...]:
    """Recover exact custody with committed cleanup-start and removal evidence.

    Each phase rechecks ownership under its own writer transaction. Ambiguous
    intents never resolve automatically, and live foreign supervisors retain
    ownership. Three bounded phases suffice; this is not a background retry loop.
    """
    rows = manager._ledger.connection().execute("SELECT job_id FROM job_purges").fetchall()
    resolved: list[str] = []
    for row in rows:
        job_id = str(row["job_id"])
        if own_job is not None and own_job != job_id:
            continue
        for _ in range(3):
            outcome = _recover_phase(manager, job_id, own_job)
            if outcome == "resolved":
                resolved.append(job_id)
            if outcome != "advance":
                break
    return tuple(resolved)
