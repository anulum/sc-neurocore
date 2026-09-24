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

_HELD: dict[str, sqlite3.Connection] = {}
_GUARD = threading.Lock()


class StudioApiProcessConflict(RuntimeError):
    """Another process already serves this identity store."""


def api_lock_path(identity_path: Path) -> Path:
    """Return the file whose exclusive transaction marks the serving process."""
    return identity_path.with_name(f"{identity_path.name}.api-lock")


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
    """
    lock_path = api_lock_path(identity_path)
    key = str(lock_path.resolve())
    with _GUARD:
        if key in _HELD:
            return lock_path
        connection = sqlite3.connect(
            lock_path, timeout=0, isolation_level=None, check_same_thread=False
        )
        try:
            connection.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            connection.close()
            raise StudioApiProcessConflict(
                f"another Studio API process serves {identity_path.name}: browser sessions and "
                "login throttles are kept in that process, so one identity store is served by "
                "one API process (run a single server worker)"
            ) from exc
        _HELD[key] = connection
    return lock_path


__all__ = ["StudioApiProcessConflict", "api_lock_path", "hold_identity_store"]
