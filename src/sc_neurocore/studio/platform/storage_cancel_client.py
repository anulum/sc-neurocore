# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API client for storage cancellation

"""Ask the storage authority to record a cancellation; repeat on a lost reply."""

from __future__ import annotations

import secrets
import socket

from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot
from sc_neurocore.studio.platform.storage_cancel_protocol import (
    CANCEL_SCHEMA_VERSION,
    StorageCancelRequest,
    decode_cancel_response,
    encode_cancel_message,
)
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester


def cancel_request(
    workspace: str, job_id: str, *, requester: StorageRequester | None
) -> StorageCancelRequest:
    """Build a cancellation with a fresh random request ID."""
    return StorageCancelRequest(
        schema_version=CANCEL_SCHEMA_VERSION,
        operation="cancel",
        request_id=secrets.token_hex(16),
        workspace=workspace,
        requester=requester,
        job_id=job_id,
    )


def exchange_cancel(
    channel: socket.socket,
    request: StorageCancelRequest,
    *,
    expected_service_uid: int,
    max_bytes: int,
    deadline: float,
) -> StudioJobRecord:
    """Run one cancellation over a connected, exclusively owned stream.

    Returns
    -------
    StudioJobRecord
        The job's complete record after the request.

    Raises
    ------
    PermissionError
        The peer is not the configured storage identity, or policy denied.
    KeyError
        The job is not in the workspace.
    ValueError
        The reply is malformed, answers another request, or names another job.
    TimeoutError, EOFError, OSError
        The exchange failed; repeating the request is safe.
    """
    with channel:
        write_verified_frame(
            channel,
            encode_cancel_message(request),
            expected_uid=expected_service_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )
        reply = read_verified_frame(
            channel, expected_uid=expected_service_uid, max_bytes=max_bytes, deadline=deadline
        )
    response = decode_cancel_response(reply, request=request, max_bytes=max_bytes)
    # An answered cancellation always carries its record.
    record = decode_job_snapshot(response.record or {})
    if record.job_id != request.job_id or record.workspace != request.workspace:
        raise ValueError("storage cancel record does not match the request")
    return record


__all__ = ["cancel_request", "exchange_cancel"]
