# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — launcher-started worker registration grant

"""Grant task import to one launcher-started worker over a per-generation socket.

A worker started by a separate launcher is not a child of the trusted API and
cannot inherit the private registration pipe used in embedded mode. Instead the
API binds one Unix stream endpoint inside a directory it already holds, and the
worker connects to it before importing any task code. The kernel-reported peer
UID, PID and process start token must equal the identity the launcher reported
for the exact job generation. Only after the caller's registration callback has
committed that identity does the worker receive the existing ``ready`` token.
Peer verification, not the endpoint's file mode, authorises the grant.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import select
import socket
import stat
import time
from collections.abc import Callable
from types import TracebackType

from sc_neurocore.studio.platform.jobs_worker_registration import await_worker_registration
from sc_neurocore.studio.platform.storage_peer import (
    require_storage_peer,
    require_storage_supervisor_identity,
)

GRANT_ENDPOINT_NAME = "grant.sock"
_READY = b"ready\n"
_UNIX_PATH_LIMIT = 108


@dataclass(frozen=True, slots=True)
class ExpectedWorker:
    """Worker process generation reported by the trusted launcher.

    ``uid`` is the configured compute identity, ``pid`` and ``start_token``
    identify the launched process generation. The values are launcher
    observations; the endpoint compares them with kernel peer credentials.
    """

    uid: int
    pid: int
    start_token: str

    def __post_init__(self) -> None:
        """Refuse root, non-positive or malformed expectations before any accept."""
        if type(self.uid) is not int or not 0 < self.uid < 0xFFFFFFFF:
            raise ValueError("worker grant UID must be a non-root OS identity")
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("worker grant PID must be positive")
        token = self.start_token
        if not isinstance(token, str) or not token.isascii() or not token.isdecimal():
            raise ValueError("worker grant start token must be decimal")
        if token.startswith("0"):
            raise ValueError("worker grant start token must be a known generation")


def validate_grant_name(name: str) -> str:
    """Return the grant endpoint name when it is the fixed per-generation name.

    Each launch generation has its own spool directory, so the endpoint name
    itself is constant and short enough to keep the full socket path within
    the Unix limit.

    Parameters
    ----------
    name : str
        Final path component; it must equal :data:`GRANT_ENDPOINT_NAME`.

    Returns
    -------
    str
        The unchanged name.

    Raises
    ------
    ValueError
        The name has any other value.
    """
    if not isinstance(name, str) or name != GRANT_ENDPOINT_NAME:
        raise ValueError("worker grant endpoint name is invalid")
    return name


class WorkerGrantEndpoint:
    """Own one single-use grant socket inside a held API directory descriptor.

    The caller keeps ``directory_fd`` open for the endpoint's lifetime and never
    passes it to a worker. :meth:`open` refuses an existing entry, so a stale or
    substituted socket is never adopted. :meth:`close` removes only the inode it
    created. The endpoint grants at most one worker and then closes its listener.
    """

    def __init__(self, directory_fd: int, directory_path: Path, name: str) -> None:
        """Retain the held directory, its canonical path and the endpoint name."""
        self._directory_fd = directory_fd
        self._directory_path = directory_path
        self._name = validate_grant_name(name)
        self._listener: socket.socket | None = None
        self._identity: tuple[int, int] | None = None

    @property
    def path(self) -> Path:
        """Return the absolute endpoint path the worker must connect to."""
        return self._directory_path / self._name

    def _entry(self) -> os.stat_result:
        return os.stat(self._name, dir_fd=self._directory_fd, follow_symlinks=False)

    def open(self) -> None:
        """Bind and listen on a new socket inode through the held directory.

        Raises
        ------
        RuntimeError
            The endpoint is already open.
        PermissionError
            The held directory no longer names the canonical path.
        FileExistsError
            An entry with the endpoint name already exists.
        ValueError
            The endpoint path exceeds the Unix socket path limit.
        OSError
            Bind, mode change or listen fails.
        """
        if self._listener is not None:
            raise RuntimeError("worker grant endpoint is already open")
        if len(os.fsencode(str(self.path))) >= _UNIX_PATH_LIMIT:
            raise ValueError("worker grant endpoint path exceeds the Unix limit")
        held = os.fstat(self._directory_fd)
        named = os.stat(self._directory_path, follow_symlinks=False)
        if not stat.S_ISDIR(held.st_mode) or (held.st_dev, held.st_ino) != (
            named.st_dev,
            named.st_ino,
        ):
            raise PermissionError("worker grant directory changed")
        try:
            self._entry()
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("worker grant endpoint already exists")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        identity: tuple[int, int] | None = None
        try:
            listener.set_inheritable(False)
            listener.bind(f"/proc/self/fd/{self._directory_fd}/{self._name}")
            # The held directory is API-owned: only this identity can replace
            # the entry just bound, so its inode is recorded without re-proving it.
            created = self._entry()
            identity = (created.st_dev, created.st_ino)
            # Connecting needs write permission on the socket. Reach is gated by the
            # API-owned generation directory; authorisation is kernel peer checks.
            os.chmod(  # nosec B103
                self._name, 0o666, dir_fd=self._directory_fd, follow_symlinks=False
            )
            listener.listen(4)
        except BaseException:
            listener.close()
            if identity is not None:
                self._unlink_own(identity)
            raise
        self._listener = listener
        self._identity = identity

    def _unlink_own(self, identity: tuple[int, int]) -> None:
        try:
            current = self._entry()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) == identity and stat.S_ISSOCK(current.st_mode):
            os.unlink(self._name, dir_fd=self._directory_fd)

    def grant(
        self,
        expected: ExpectedWorker,
        *,
        deadline: float,
        max_refusals: int,
        register: Callable[[str], None],
    ) -> str:
        """Verify the launched worker, commit its identity and send one grant.

        Parameters
        ----------
        expected : ExpectedWorker
            Launcher-reported compute UID, PID and start token.
        deadline : float
            Absolute monotonic deadline for accepting and granting.
        max_refusals : int
            Positive number of unverified connections tolerated before refusing.
        register : Callable[[str], None]
            Durable registration of the verified ``host:pid:token`` identity.
            It runs after verification and before ``ready`` is written.

        Returns
        -------
        str
            The verified worker identity that received the grant.

        Raises
        ------
        RuntimeError
            The endpoint is not open.
        ValueError
            The refusal budget is not a positive integer.
        TimeoutError
            No verified worker connected and received the grant in time.
        PermissionError
            More unverified connections arrived than the refusal budget allows.
        OSError
            Writing the grant failed after registration committed, for example
            because the worker already closed; registration is not rolled back.
        Exception
            A registration failure propagates after the worker connection is
            closed without a grant; the worker then refuses task import.

        Notes
        -----
        The listener is closed on every outcome, so a single endpoint never
        grants a second worker or a later generation. A written ``ready`` is
        not proof that the worker accepted it: the worker may refuse the
        server identity or exit first. The supervisor owns that observation.
        """
        listener = self._listener
        if listener is None:
            raise RuntimeError("worker grant endpoint is not open")
        if type(max_refusals) is not int or max_refusals <= 0:
            raise ValueError("worker grant refusal budget must be positive")
        refusals = 0
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([listener], [], [], remaining)[0]:
                    raise TimeoutError("launched worker did not request its grant in time")
                channel, _ = listener.accept()
                with channel:
                    try:
                        identity = require_storage_supervisor_identity(
                            channel, expected_uid=expected.uid
                        )
                    except PermissionError:
                        identity = None
                    _, pid_text, token = (
                        ("", "", "") if identity is None else identity.split(":", 2)
                    )
                    if (
                        identity is None
                        or pid_text != str(expected.pid)
                        or token != expected.start_token
                    ):
                        refusals += 1
                        if refusals > max_refusals:
                            raise PermissionError("worker grant refused unverified connections")
                        continue
                    register(identity)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("worker grant deadline expired before ready")
                    channel.settimeout(remaining)
                    channel.sendall(_READY)
                    channel.shutdown(socket.SHUT_WR)
                    return identity
        finally:
            self.close()

    def close(self) -> None:
        """Close the listener and remove only this endpoint's unchanged socket."""
        listener = self._listener
        identity = self._identity
        self._listener = None
        self._identity = None
        if listener is None or identity is None:
            return
        listener.close()
        self._unlink_own(identity)

    def __enter__(self) -> WorkerGrantEndpoint:
        """Open the endpoint and return its single owner."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the endpoint without suppressing caller failures."""
        self.close()


def receive_socket_grant(path: Path, *, expected_server_uid: int) -> None:
    """Connect to the API grant endpoint and require the exact ``ready`` grant.

    Parameters
    ----------
    path : Path
        Absolute endpoint path supplied by the trusted launcher descriptor.
    expected_server_uid : int
        Configured API identity that must own the accepting socket.

    Raises
    ------
    ValueError
        The path is relative or its name does not match the grant grammar.
    PermissionError
        The connected server is not the configured API identity.
    RuntimeError
        EOF, a malformed token, trailing bytes or the three-second reader
        deadline refuses the grant; no task may be imported.
    OSError
        The endpoint is missing or refuses the connection.
    """
    if not path.is_absolute():
        raise ValueError("worker grant endpoint path must be absolute")
    validate_grant_name(path.name)
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with channel:
        channel.settimeout(3.0)
        channel.connect(str(path))
        require_storage_peer(channel, expected_uid=expected_server_uid)
        channel.setblocking(True)
        await_worker_registration(channel.fileno())
