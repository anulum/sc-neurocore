# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — One Studio API process per identity store

"""Hold an identity store for one Studio API process.

Browser sessions and login throttles live in the API process's memory. A
second API process serving the same identity store — a second server worker,
or a second server started on the same files — would hold its own sessions and
its own failure counts: a session signed out in one stays valid in the other,
and a password guess limit multiplies by the number of processes. The Studio's
supported deployment is one lab process, so that boundary is enforced rather
than assumed: the first API process to open an identity store holds it for its
lifetime, and any other process that tries is refused with the reason.

The hold is an exclusive SQLite transaction on a file beside the store, the
same portable locking the workspace store uses. It is released when the process
ends, however it ends. Opening the store again in the process that holds it
succeeds, so one process may build its application more than once.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_HELD: dict[tuple[str, int, int], tuple[sqlite3.Connection, ...]] = {}
_GUARD = threading.Lock()


class StudioApiProcessConflict(RuntimeError):
    """Another process already serves this identity store."""


class StudioApiLockUnavailable(RuntimeError):
    """The lock file beside the identity store cannot be created or opened."""


def api_lock_path(identity_path: Path) -> Path:
    """Return the file whose exclusive transaction marks the serving process."""
    canonical = identity_path.expanduser().resolve(strict=True)
    return canonical.with_name(f"{canonical.name}.api-lock")


def hold_identity_store(identity_path: Path) -> Path:
    """Hold ``identity_path`` for this process, or refuse because another holds it.

    Returns
    -------
    pathlib.Path
        The lock file.

    Raises
    ------
    StudioApiProcessConflict
        When another process holds the store.
    StudioApiLockUnavailable
        When the lock file cannot be created or opened, for example because
        the identity store's directory is not writable by this account.
    """
    canonical = identity_path.expanduser().resolve(strict=True)
    info = canonical.stat()
    lock_path = api_lock_path(canonical)
    inode_lock = canonical.parent / f".studio-api-{info.st_dev:x}-{info.st_ino:x}.lock"
    key = (str(canonical), info.st_dev, info.st_ino)
    with _GUARD:
        # Keep the path lock across atomic file replacements as well as the
        # original inode hold. Parent/symlink aliases converge before locking.
        if any(held[0] == key[0] for held in _HELD):
            return lock_path
        opened: list[sqlite3.Connection] = []
        try:
            for path in (lock_path, inode_lock):
                connection = sqlite3.connect(
                    path, timeout=0, isolation_level=None, check_same_thread=False
                )
                opened.append(connection)
                connection.execute("PRAGMA journal_mode=MEMORY")
                connection.execute("BEGIN EXCLUSIVE")
        except sqlite3.DatabaseError as exc:
            for connection in opened:
                connection.close()
            if "locked" not in str(exc):
                raise _lock_unavailable(lock_path, exc) from exc
            raise StudioApiProcessConflict(
                f"another Studio API process serves {identity_path.name}: browser sessions and "
                "login throttles are kept in that process, so one identity store is served by "
                "one API process (run a single server worker)"
            ) from exc
        _HELD[key] = tuple(opened)
    return lock_path


def _lock_unavailable(lock_path: Path, exc: sqlite3.DatabaseError) -> StudioApiLockUnavailable:
    return StudioApiLockUnavailable(
        f"the Studio cannot open its API lock {lock_path} ({exc}); the lock is a file beside "
        "the identity store that the account running Studio must be able to create and open"
    )


__all__ = [
    "StudioApiLockUnavailable",
    "StudioApiProcessConflict",
    "api_lock_path",
    "hold_identity_store",
]
