# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Peer-verified storage record client

"""Read through the existing connected authority without opening local storage."""

import socket
import time

from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_connection import connect_storage_authority
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_record_protocol import (
    StorageRecordRequest,
    decode_record_response,
)


def read_storage_record(
    channel: socket.socket,
    *,
    request: StorageRecordRequest,
    expected_service_uid: int,
    max_bytes: int,
    deadline: float,
) -> StudioJobRecord:
    """Exchange one bounded request with a peer-verified storage authority.

    Parameters
    ----------
    channel : socket.socket
        Connected Unix stream, exclusively owned and closed on every outcome.
    request : StorageRecordRequest
        Exact validated read request. Requester claims must come from the trusted
        API authentication adapter, never directly from a compute worker.
    expected_service_uid : int
        Service OS identity supplied by trusted configuration, not the peer.
    max_bytes : int
        Positive uint32 ceiling for each complete request and response frame.
    deadline : float
        Absolute monotonic deadline shared by both transfers, never renewed.

    Returns
    -------
    StudioJobRecord
        Complete correlated native snapshot, without creating a local ledger.

    Raises
    ------
    ValueError
        Configuration, framing, schema, correlation or snapshot is invalid.
    PermissionError
        Service peer identity or authority policy refuses the operation.
    KeyError
        Authority reports no record in the configured workspace.
    TimeoutError
        The transfer deadline expires.
    EOFError
        The authority disconnects before a complete response arrives.
    OSError
        Socket transfer fails.

    Notes
    -----
    No reconnect, retry, mutation, local fallback or isolated readiness claim.
    Connection establishment and protected endpoint configuration belong to the
    runtime lifecycle owner; this function consumes an already connected socket.
    """
    with channel:
        write_verified_frame(
            channel,
            request.model_dump_json().encode("utf-8"),
            expected_uid=expected_service_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )
        response = read_verified_frame(
            channel,
            expected_uid=expected_service_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )
        return decode_record_response(response, request=request)


def read_storage_record_at_endpoint(
    configuration: StorageBoundaryConfiguration,
    *,
    request: StorageRecordRequest,
) -> StudioJobRecord:
    """Read one record over the configured, peer-verified service endpoint.

    The caller must construct ``request.requester`` from the API's authenticated
    principal. The server independently checks policy and binds the workspace.
    A failed connection or read has no local ledger fallback and no retry.
    Connection and frame exchange each have the configured finite timeout.
    """
    if request.workspace != configuration.workspace:
        raise ValueError("storage request does not match configured workspace")
    channel = connect_storage_authority(configuration)
    return read_storage_record(
        channel,
        request=request,
        expected_service_uid=configuration.storage_uid,
        max_bytes=configuration.frame_max_bytes,
        deadline=time.monotonic() + configuration.transfer_timeout_seconds,
    )
