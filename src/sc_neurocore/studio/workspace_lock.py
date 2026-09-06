# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — One writer at a time per Studio workspace

"""One writer at a time per workspace, across threads and across processes.

An append-only revision store refuses a stale save only if reading the head
and writing the next revision happen together. They did not: two savers read
the same head, both believed they were creating revision 1, and both were
acknowledged while one payload was overwritten. A check that another writer
can walk through is not a check.

The exclusion is a SQLite ``BEGIN IMMEDIATE`` on a small database of its own,
one per workspace, which is the same primitive the job ledger already relies
on for single-host serialisation:

* it is enforced by the operating system, so **separate processes** sharing
  one project root exclude each other, not only threads of one server;
* it is released when the connection closes **and when the process dies**, so
  a crashed or killed worker cannot leave a workspace permanently locked;
* the wait is **bounded**: a writer that cannot take the lock within the
  timeout is refused with :class:`WorkspaceLockTimeout` rather than blocking
  a request thread indefinitely.

The lock database lives beside the workspaces rather than inside one, because
deleting a workspace moves its directory into the trash: a lock kept inside
would vanish underneath the writer holding it, and restoring would bring back
a stale one.

Within one process the lock is re-entrant per thread, so a locked operation
may call another one — a save reads the head, a restore checks for a live
workspace — without deadlocking against itself. A different thread still
waits, and so does a different process.

Limitation, stated rather than implied: this serialises writers that share a
filesystem which implements SQLite's locking. Two hosts over one network
share are outside what it can promise, exactly as for the job ledger.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

#: Directory under the project root holding one lock database per workspace.
LOCK_DIR = ".locks"
#: Suffix of one workspace's lock database.
LOCK_SUFFIX = ".lock.sqlite3"
#: How long a writer waits for another writer before being refused, in seconds.
DEFAULT_LOCK_TIMEOUT = 10.0


class WorkspaceLockTimeout(TimeoutError):
    """Raised when another writer held a workspace for longer than the wait.

    Attributes
    ----------
    name : str
        The workspace that was busy.
    timeout : float
        The wait that elapsed, in seconds.
    """

    def __init__(self, *, name: str, timeout: float) -> None:
        super().__init__(
            f"another writer held workspace {name!r} for longer than {timeout} seconds; "
            "nothing was written, so the same save can be retried."
        )
        self.name = name
        self.timeout = timeout

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {
            "error": "workspace_busy",
            "name": self.name,
            "reason": str(self),
            "timeout_seconds": self.timeout,
        }


def lock_path(root: Path, name: str) -> Path:
    """Return the lock database of one workspace under a project root.

    Parameters
    ----------
    root : pathlib.Path
        The project root holding the workspaces.
    name : str
        Workspace name, one path segment.

    Raises
    ------
    ValueError
        The name is not a single path segment. The callers validate names
        already; this refuses rather than trusting them, because the result is
        a filesystem path.
    """
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("Invalid workspace name")
    return root / LOCK_DIR / f"{name}{LOCK_SUFFIX}"


@dataclass
class _Holding:
    """One workspace's lock as this process holds it.

    ``depth`` counts the nested acquisitions of the thread that holds
    ``guard``, so only the outermost one opens and closes the database
    connection that excludes the other processes.
    """

    guard: threading.RLock = field(default_factory=threading.RLock)
    depth: int = 0


_REGISTRY_GUARD = threading.Lock()
_REGISTRY: dict[str, _Holding] = {}


def _holding(path: Path) -> _Holding:
    """Return this process's record for one lock database, creating it once."""
    key = str(path)
    with _REGISTRY_GUARD:
        holding = _REGISTRY.get(key)
        if holding is None:
            holding = _Holding()
            _REGISTRY[key] = holding
        return holding


def _connect(path: Path, name: str, timeout: float) -> sqlite3.Connection:
    """Open the lock database and take its write lock, or refuse in time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=max(timeout, 0.0), isolation_level=None)
    try:
        connection.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as exc:
        connection.close()
        raise WorkspaceLockTimeout(name=name, timeout=timeout) from exc
    except BaseException:
        connection.close()
        raise
    return connection


@contextmanager
def workspace_lock(
    root: Path, name: str, *, timeout: float = DEFAULT_LOCK_TIMEOUT
) -> Iterator[None]:
    """Hold one workspace against every other writer for the block's duration.

    Parameters
    ----------
    root : pathlib.Path
        The project root holding the workspaces.
    name : str
        Workspace name.
    timeout : float, optional
        How long to wait for another writer before refusing.

    Yields
    ------
    None
        The block runs with the workspace held.

    Raises
    ------
    WorkspaceLockTimeout
        Another thread or process held the workspace for the whole wait.
        Nothing was written.
    ValueError
        The name is not a single path segment.
    """
    path = lock_path(root, name)
    holding = _holding(path)
    started = time.monotonic()
    if not holding.guard.acquire(timeout=max(timeout, 0.0)):
        raise WorkspaceLockTimeout(name=name, timeout=timeout)
    opened: sqlite3.Connection | None = None
    try:
        if holding.depth == 0:
            remaining = timeout - (time.monotonic() - started)
            opened = _connect(path, name, remaining)
        holding.depth += 1
    except BaseException:
        holding.guard.release()
        raise
    try:
        yield
    finally:
        holding.depth -= 1
        if opened is not None:
            # Only the outermost acquisition closes it. Closing rolls the
            # transaction back and releases the write lock, whether the block
            # returned or raised.
            opened.close()
        holding.guard.release()


__all__ = [
    "DEFAULT_LOCK_TIMEOUT",
    "LOCK_DIR",
    "LOCK_SUFFIX",
    "WorkspaceLockTimeout",
    "lock_path",
    "workspace_lock",
]
