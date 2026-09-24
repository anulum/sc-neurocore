# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — bounded storage wire framing

"""Bounded byte framing, without authentication or storage dispatch.

Callers exclusively own an already-connected Unix stream socket and authenticate
its peer separately. A transfer error closes the ambiguous stream; successful
operations restore its previous timeout. There is no automatic replay.
"""

from __future__ import annotations

import math
import socket
import struct
from threading import TIMEOUT_MAX
import time

_HEADER = struct.Struct("!I")


def _validate(channel: socket.socket, max_bytes: int, deadline: float) -> None:
    if type(max_bytes) is not int or not 0 < max_bytes <= 0xFFFFFFFF:
        raise ValueError("max_bytes must be a positive uint32 integer")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise ValueError("deadline must be finite monotonic seconds")
    try:
        finite = math.isfinite(deadline)
    except OverflowError as exc:
        raise ValueError("deadline must be representable finite monotonic seconds") from exc
    if not finite:
        raise ValueError("deadline must be finite monotonic seconds")
    if deadline - time.monotonic() > TIMEOUT_MAX:
        raise ValueError("deadline exceeds the platform timeout range")
    if channel.family != socket.AF_UNIX:
        raise ValueError("storage framing requires a Unix socket")
    if channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM:
        raise ValueError("storage framing requires a stream socket")


def _remaining(channel: socket.socket, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("storage frame deadline expired")
    channel.settimeout(remaining)


def _receive(channel: socket.socket, size: int, deadline: float) -> bytes:
    buffer = bytearray(size)
    offset = 0
    while offset < size:
        _remaining(channel, deadline)
        count = channel.recv_into(memoryview(buffer)[offset:])
        if count == 0:
            raise EOFError("peer closed during storage frame")
        offset += count
    return bytes(buffer)


def read_frame(channel: socket.socket, *, max_bytes: int, deadline: float) -> bytes:
    """Receive one nonempty frame within an absolute deadline.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned connected Unix stream; peer authentication is external.
    max_bytes : int
        Maximum permitted payload bytes, from one through uint32 maximum.
    deadline : float
        Absolute monotonic time in seconds, shared by header and payload reads.
        The remaining interval must fit the platform timeout range.

    Returns
    -------
    bytes
        Complete payload; the four-byte network-order header is not returned.

    Raises
    ------
    ValueError
        Invalid arguments or an empty/oversized declared payload.
    TimeoutError
        The absolute deadline expires.
    EOFError
        The peer closes before the complete frame arrives.
    OSError
        Socket transfer fails. Transfer failures close the connection.
    """
    _validate(channel, max_bytes, deadline)
    previous = channel.gettimeout()
    try:
        size = _HEADER.unpack(_receive(channel, _HEADER.size, deadline))[0]
        if not 0 < size <= max_bytes:
            raise ValueError("declared storage frame size exceeds permitted bounds")
        payload = _receive(channel, size, deadline)
        channel.settimeout(previous)
        return payload
    except BaseException:
        channel.close()
        raise


def write_frame(channel: socket.socket, payload: bytes, *, max_bytes: int, deadline: float) -> None:
    """Send one nonempty byte payload under a single absolute deadline.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned connected Unix stream; peer authentication is external.
    payload : bytes
        Nonempty immutable payload, excluding the generated length header.
    max_bytes : int
        Maximum payload bytes, from one through uint32 maximum.
    deadline : float
        Finite absolute monotonic time in seconds for the entire transfer.
        The remaining interval must fit the platform timeout range.

    Raises
    ------
    ValueError
        Arguments or payload type/size are invalid; no transfer is attempted.
    TimeoutError
        The total transfer deadline expires, including sender backpressure.
    OSError
        Socket transfer fails. Transfer failures close the connection.

    Notes
    -----
    Header and body use separate sends to avoid a combined payload allocation.
    Success restores the caller's timeout. Failure never retries a mutation.
    """
    _validate(channel, max_bytes, deadline)
    if not isinstance(payload, bytes) or not 0 < len(payload) <= max_bytes:
        raise ValueError("payload must be nonempty bytes within max_bytes")
    previous = channel.gettimeout()
    try:
        _remaining(channel, deadline)
        channel.sendall(_HEADER.pack(len(payload)))
        _remaining(channel, deadline)
        channel.sendall(payload)
        channel.settimeout(previous)
    except BaseException:
        channel.close()
        raise
