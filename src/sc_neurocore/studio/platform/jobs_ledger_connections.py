# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job ledger connection registry

"""Every sqlite connection a job ledger opens, so that all of them can be closed.

The ledger gives each thread its own connection. Before this registry, closing
the ledger closed only the calling thread's connection; the others were freed
by their finalizers at an unpredictable later point, which on Python 3.13 and
later raises ``ResourceWarning: unclosed database`` in whatever code happens to
run then.
"""

from __future__ import annotations

import sqlite3
import threading
import weakref
from pathlib import Path


class LedgerConnectionRegistry:
    """Thread-owned ledger connections, each closed only when nothing can still use it.

    Each connection is used only by the thread that opened it. A connection is
    closed by its own thread, or by another thread only once its owner has
    ended; a live thread may be inside a transaction, and closing its
    connection under it would crash the interpreter. Connections are created
    with ``check_same_thread=False`` only so that an ended thread's connection
    can be closed from another thread.
    """

    def __init__(self, *, owner: object) -> None:
        """Track connections for ``owner``; all are closed when ``owner`` is collected."""
        self._lock = threading.Lock()
        self._connections: list[tuple[threading.Thread, sqlite3.Connection]] = []
        self._generation = 0
        # The callback holds this registry, never the owner, so it cannot keep the owner alive.
        # Once the owner is unreachable no thread can be using its connections.
        weakref.finalize(owner, self._close_every)

    @property
    def generation(self) -> int:
        """Return how many times :meth:`close_all` has run.

        A thread whose cached connection belongs to an earlier generation must
        close it with :meth:`forget` and open a new one.
        """
        return self._generation

    def current(
        self, entry: tuple[int, sqlite3.Connection] | None, path: Path, busy_timeout_ms: int
    ) -> tuple[int, sqlite3.Connection]:
        """Return the calling thread's ``(generation, connection)``, opening it if needed.

        ``entry`` is what the caller cached. A connection retired by
        :meth:`close_all` is replaced here, by its own thread; while it is still
        inside a transaction it is kept, because closing it would abort that
        transaction. It is replaced on the first use after the transaction ends.
        """
        if entry is not None and (entry[0] == self._generation or entry[1].in_transaction):
            return entry
        retired = None if entry is None else entry[1]
        return self._generation, self.open(path, busy_timeout_ms, retired)

    def open(
        self, path: Path, busy_timeout_ms: int, retired: sqlite3.Connection | None = None
    ) -> sqlite3.Connection:
        """Open a ledger connection for the calling thread and record it.

        The connection runs in autocommit mode (transactions are explicit), in
        WAL journal mode with full synchronous writes, and waits up to
        ``busy_timeout_ms`` for a lock. ``retired`` is the caller's own
        connection from an earlier generation; it is closed first, as are the
        connections of threads that have ended.
        """
        if retired is not None:
            self.forget(retired)
        connection = sqlite3.connect(
            path,
            timeout=busy_timeout_ms / 1000.0,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        with self._lock:
            stale = [c for t, c in self._connections if not t.is_alive()]
            self._connections = [(t, c) for t, c in self._connections if t.is_alive()]
            self._connections.append((threading.current_thread(), connection))
        for old in stale:
            old.close()
        return connection

    def forget(self, connection: sqlite3.Connection) -> None:
        """Close ``connection`` and stop tracking it."""
        with self._lock:
            self._connections = [(t, c) for t, c in self._connections if c is not connection]
        connection.close()

    def close_all(self) -> None:
        """Close the caller's and every ended thread's connection; retire the rest.

        A connection of another live thread is left open, because that thread
        may be using it right now. It belongs to an earlier generation from here
        on, so its owner closes it on its next use of the ledger.
        """
        me = threading.current_thread()
        with self._lock:
            self._generation += 1
            closable = [c for t, c in self._connections if t is me or not t.is_alive()]
            self._connections = [
                (t, c) for t, c in self._connections if t is not me and t.is_alive()
            ]
        for connection in closable:
            connection.close()

    def _close_every(self) -> None:
        with self._lock:
            connections = [c for _, c in self._connections]
            self._connections = []
        for connection in connections:
            connection.close()

    def __len__(self) -> int:
        """Return how many connections are open and recorded."""
        with self._lock:
            return len(self._connections)
