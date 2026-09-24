# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API client for delegated job supervision

"""Ask the storage authority to start or renew a job the calling API owns.

A timeout or disconnect after sending ``start`` is ambiguous: the authority
may have registered the worker. Repeating ``start`` with the same worker
identity is safe; the authority answers ``started`` for an already registered
identical worker and ``worker_conflict`` for a different one.
"""

from __future__ import annotations

import secrets
import socket
import time

from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_connection import connect_storage_authority
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_supervision_protocol import (
    SUPERVISION_SCHEMA_VERSION,
    StorageSupervisionResponse,
    SupervisionHeartbeatRequest,
    SupervisionStartRequest,
    decode_supervision_response,
    encode_supervision_message,
)


def supervision_start_request(
    configuration: StorageBoundaryConfiguration, *, job_id: str, worker: str
) -> SupervisionStartRequest:
    """Build a ``start`` request for the configured workspace.

    Parameters
    ----------
    configuration : StorageBoundaryConfiguration
        Trusted boundary configuration supplying the workspace.
    job_id : str
        Admitted job ID.
    worker : str
        ``host:pid:token`` identity verified at the grant endpoint.

    Returns
    -------
    SupervisionStartRequest
        Validated request with a fresh random request ID.
    """
    return SupervisionStartRequest(
        schema_version=SUPERVISION_SCHEMA_VERSION,
        operation="start",
        request_id=secrets.token_hex(16),
        workspace=configuration.workspace,
        job_id=job_id,
        worker=worker,
    )


def supervision_heartbeat_request(
    configuration: StorageBoundaryConfiguration, *, job_id: str
) -> SupervisionHeartbeatRequest:
    """Build a ``heartbeat`` request for the configured workspace.

    Parameters
    ----------
    configuration : StorageBoundaryConfiguration
        Trusted boundary configuration supplying the workspace.
    job_id : str
        Job whose delegated lease should be renewed.

    Returns
    -------
    SupervisionHeartbeatRequest
        Validated request with a fresh random request ID.
    """
    return SupervisionHeartbeatRequest(
        schema_version=SUPERVISION_SCHEMA_VERSION,
        operation="heartbeat",
        request_id=secrets.token_hex(16),
        workspace=configuration.workspace,
        job_id=job_id,
    )


def exchange_supervision(
    channel: socket.socket,
    request: SupervisionStartRequest | SupervisionHeartbeatRequest,
    *,
    expected_service_uid: int,
    max_bytes: int,
    deadline: float,
) -> StorageSupervisionResponse:
    """Exchange one request over a connected, exclusively owned stream.

    Parameters
    ----------
    channel : socket.socket
        Connected Unix stream to the storage authority, closed on every outcome.
    request : SupervisionStartRequest or SupervisionHeartbeatRequest
        Request to send.
    expected_service_uid : int
        Configured storage identity, checked before each frame.
    max_bytes : int
        Frame ceiling for request and response.
    deadline : float
        Absolute monotonic deadline shared by both transfers.

    Returns
    -------
    StorageSupervisionResponse
        Outcome correlated with ``request``.

    Raises
    ------
    ValueError
        The reply is malformed or answers another request.
    PermissionError
        The peer is not the configured storage identity.
    TimeoutError, EOFError, OSError
        The exchange failed; whether the authority acted is unknown.
    """
    with channel:
        write_verified_frame(
            channel,
            encode_supervision_message(request),
            expected_uid=expected_service_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )
        reply = read_verified_frame(
            channel, expected_uid=expected_service_uid, max_bytes=max_bytes, deadline=deadline
        )
    return decode_supervision_response(reply, request=request, max_bytes=max_bytes)


def exchange_supervision_request(
    configuration: StorageBoundaryConfiguration,
    request: SupervisionStartRequest | SupervisionHeartbeatRequest,
) -> StorageSupervisionResponse:
    """Send one request to the configured, peer-verified storage endpoint.

    Parameters
    ----------
    configuration : StorageBoundaryConfiguration
        Trusted endpoint, identities, frame ceiling and transfer timeout.
    request : SupervisionStartRequest or SupervisionHeartbeatRequest
        Request for the configured workspace.

    Returns
    -------
    StorageSupervisionResponse
        Outcome correlated with ``request``.

    Raises
    ------
    ValueError
        The request names another workspace, or the reply is malformed.
    PermissionError
        This process is not the configured API identity, or the service peer
        is not the configured storage identity.
    TimeoutError, EOFError, OSError
        The exchange failed; whether the authority acted is unknown.
    """
    if request.workspace != configuration.workspace:
        raise ValueError("storage request does not match configured workspace")
    return exchange_supervision(
        connect_storage_authority(configuration),
        request,
        expected_service_uid=configuration.storage_uid,
        max_bytes=configuration.frame_max_bytes,
        deadline=time.monotonic() + configuration.transfer_timeout_seconds,
    )
