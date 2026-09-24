# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API client for the worker launcher

"""Send one launcher request from the trusted API and read its correlated reply.

The API verifies that the configured launcher identity owns the connected
endpoint before sending anything. A timeout or disconnect after the request
was sent is ambiguous: the worker may or may not have started. The caller
resolves it by sending ``status`` for the same job generation, and it retries
``launch`` only with that same generation, which the launcher answers from
its record instead of starting a second worker.
"""

from __future__ import annotations

from pathlib import Path
import secrets
import socket
import time

from sc_neurocore.studio.platform.storage_launcher_protocol import (
    LAUNCHER_MESSAGE_MAX_BYTES,
    LAUNCHER_PROTOCOL_VERSION,
    LauncherOperation,
    LauncherRequest,
    LauncherResponse,
    decode_launcher_response,
    encode_launcher_request,
)
from sc_neurocore.studio.platform.storage_peer import require_storage_peer
from sc_neurocore.studio.platform.storage_transport import read_frame, write_frame


def new_launcher_request(
    operation: LauncherOperation, *, job_id: str, generation: str
) -> LauncherRequest:
    """Build a request with a fresh random request ID.

    Parameters
    ----------
    operation : {"launch", "stop", "status"}
        Requested launcher operation.
    job_id : str
        Admitted job ID, ``sj_`` plus 16 lowercase hexadecimal digits.
    generation : str
        Launch generation chosen once per attempt, 32 lowercase hex digits.

    Returns
    -------
    LauncherRequest
        Validated request.

    Raises
    ------
    pydantic.ValidationError
        An identifier does not match the wire grammar.
    """
    return LauncherRequest(
        version=LAUNCHER_PROTOCOL_VERSION,
        request_id=secrets.token_hex(16),
        operation=operation,
        job_id=job_id,
        generation=generation,
    )


def exchange_launcher_request(
    socket_path: Path,
    request: LauncherRequest,
    *,
    launcher_uid: int,
    deadline: float,
) -> LauncherResponse:
    """Deliver one request to the verified launcher and return its reply.

    Parameters
    ----------
    socket_path : Path
        Configured launcher endpoint.
    request : LauncherRequest
        Request to send.
    launcher_uid : int
        Configured launcher identity that must own the accepting socket.
    deadline : float
        Absolute monotonic deadline for connect, send and receive.

    Returns
    -------
    LauncherResponse
        Reply correlated with ``request``.

    Raises
    ------
    PermissionError
        The endpoint is not owned by the configured launcher identity; no
        request bytes were sent.
    TimeoutError
        The deadline expired; whether the launcher acted is unknown.
    EOFError
        The launcher closed without a reply; whether it acted is unknown.
    ValueError
        The reply is malformed or does not answer this request.
    OSError
        The endpoint is missing or the transfer failed.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("launcher request deadline already expired")
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with channel:
        channel.settimeout(remaining)
        channel.connect(str(socket_path))
        require_storage_peer(channel, expected_uid=launcher_uid)
        write_frame(
            channel,
            encode_launcher_request(request),
            max_bytes=LAUNCHER_MESSAGE_MAX_BYTES,
            deadline=deadline,
        )
        reply = read_frame(channel, max_bytes=LAUNCHER_MESSAGE_MAX_BYTES, deadline=deadline)
    return decode_launcher_response(reply, request=request)
