# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — connected storage peer verification

"""Verify connected Linux peer identity before bounded storage byte transfer.

OS credentials do not grant browser roles or workspace authority. Callers own
the socket exclusively and must not delegate its descriptor to compute workers.
Separate UID deployment and application authorization remain external contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import select
import socket
import struct
import sys

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_transport import read_frame, write_frame

_UCRED = struct.Struct("iII")
_PIDFD = struct.Struct("i")
# Linux UAPI asm-generic/socket.h; Python does not expose this option yet.
_SO_PEERPIDFD = 77


@dataclass(frozen=True, slots=True)
class StoragePeer:
    """Linux PID/UID/GID at connection creation, not live process custody.

    The PID is not a process generation token. UID/GID are OS identities, not
    authenticated HTTP principals or a workspace membership assertion.
    """

    pid: int
    uid: int
    gid: int


def require_storage_peer(channel: socket.socket, *, expected_uid: int) -> StoragePeer:
    """Require an exact configured UID on a connected Linux Unix stream.

    Parameters
    ----------
    channel : socket.socket
        Connected, exclusively owned socket. Peer rejection closes it.
    expected_uid : int
        Trusted configuration value, never a peer-supplied assertion. Valid OS
        UID from zero through uint32 maximum minus one; booleans are invalid.

    Returns
    -------
    StoragePeer
        Kernel-reported connection credentials. No liveness guarantee is implied.

    Raises
    ------
    ValueError
        Expected UID is invalid; no socket operations are attempted.
    PermissionError
        The platform/transport/peer is unsupported, unknown or not permitted.
        The rejected socket is closed without transferring framed bytes.
    """
    if type(expected_uid) is not int or not 0 <= expected_uid < 0xFFFFFFFF:
        raise ValueError("expected_uid must be a valid integer OS UID")
    try:
        if sys.platform != "linux" or channel.family != socket.AF_UNIX:
            raise PermissionError("storage peer verification requires Linux Unix transport")
        if channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM:
            raise PermissionError("storage peer verification requires a stream")
        raw = channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED.size)
        pid, uid, gid = _UCRED.unpack(raw)
        if pid <= 0 or uid == 0xFFFFFFFF or gid == 0xFFFFFFFF:
            raise PermissionError("storage peer credentials are unavailable")
        if uid != expected_uid:
            raise PermissionError("storage peer UID is not permitted")
        return StoragePeer(pid=pid, uid=uid, gid=gid)
    except (OSError, struct.error, AttributeError) as exc:
        channel.close()
        raise PermissionError("storage peer verification refused") from exc


def require_storage_supervisor_identity(channel: socket.socket, *, expected_uid: int) -> str:
    """Bind a connected trusted API peer to its kernel-anchored process generation.

    The peer pidfd must still refer to the SO_PEERCRED PID while its proc start
    token is read. An exited peer, unsupported pidfd option or unavailable
    start token refuses before admission. The returned identity is suitable
    for the existing ledger lease, not a browser authorization decision.
    """
    peer = require_storage_peer(channel, expected_uid=expected_uid)
    pidfd: int | None = None
    try:
        pidfd = _PIDFD.unpack(channel.getsockopt(socket.SOL_SOCKET, _SO_PEERPIDFD, _PIDFD.size))[0]
        if pidfd < 0:
            raise PermissionError("storage peer pidfd is unavailable")
        os.set_inheritable(pidfd, False)
        watcher = select.poll()
        watcher.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        if watcher.poll(0):
            raise PermissionError("storage supervisor peer has exited")
        with open(f"/proc/self/fdinfo/{pidfd}", encoding="ascii") as handle:
            pid_lines = [line for line in handle if line.startswith("Pid:")]
        if len(pid_lines) != 1 or int(pid_lines[0].split(":", 1)[1]) != peer.pid:
            raise PermissionError("storage peer pidfd identity changed")
        identity = supervisor_identity(peer.pid)
        token = identity.rsplit(":", 1)[-1]
        if not token.isascii() or not token.isdecimal() or token.startswith("0"):
            raise PermissionError("storage supervisor generation is unavailable")
        if watcher.poll(0):
            raise PermissionError("storage supervisor peer exited during verification")
        return identity
    except (OSError, ValueError, struct.error, AttributeError) as exc:
        channel.close()
        raise PermissionError("storage supervisor identity verification refused") from exc
    finally:
        if pidfd is not None and pidfd >= 0:
            os.close(pidfd)


def read_verified_frame(
    channel: socket.socket, *, expected_uid: int, max_bytes: int, deadline: float
) -> bytes:
    """Verify the OS peer before receiving a bounded storage frame.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned connected Unix stream.
    expected_uid : int
        Trusted peer UID; not an application role or principal assertion.
    max_bytes : int
        Positive uint32 payload ceiling passed to the existing framing owner.
    deadline : float
        Absolute monotonic deadline, not renewed by peer verification.

    Returns
    -------
    bytes
        Complete nonempty payload from the verified connection.

    Raises
    ------
    PermissionError
        Peer verification refuses before any frame read.
    ValueError
        UID, frame limit, deadline or declared length is invalid.
    TimeoutError
        The original deadline expires.
    EOFError
        The peer closes before a complete frame arrives.
    OSError
        Frame transfer fails; the ambiguous socket is closed.
    """
    require_storage_peer(channel, expected_uid=expected_uid)
    return read_frame(channel, max_bytes=max_bytes, deadline=deadline)


def write_verified_frame(
    channel: socket.socket, payload: bytes, *, expected_uid: int, max_bytes: int, deadline: float
) -> None:
    """Verify the OS peer before sending a bounded storage frame.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned connected Unix stream.
    payload : bytes
        Nonempty payload within the explicit byte ceiling.
    expected_uid : int
        Trusted peer UID, checked before sending even a length header.
    max_bytes : int
        Positive uint32 payload ceiling passed to the framing owner.
    deadline : float
        Absolute monotonic deadline, not renewed by peer verification.

    Raises
    ------
    PermissionError
        Peer verification refuses before any frame write.
    ValueError
        UID, payload, frame limit or deadline is invalid.
    TimeoutError
        The original deadline expires.
    OSError
        Frame transfer fails; the ambiguous socket is closed.
    """
    require_storage_peer(channel, expected_uid=expected_uid)
    write_frame(channel, payload, max_bytes=max_bytes, deadline=deadline)
