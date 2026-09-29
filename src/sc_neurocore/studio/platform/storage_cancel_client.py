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
from typing import Literal

from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot
from sc_neurocore.studio.platform.storage_cancel_protocol import (
    CANCEL_SCHEMA_VERSION,
    StorageCancelRequest,
    decode_cancel_response,
    encode_cancel_message,
)
from sc_neurocore.studio.platform.storage_peer import write_verified_frame
from sc_neurocore.studio.platform.storage_view_content import read_view_content, view_content_limit
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester


def cancel_request(
    workspace: str,
    job_id: str,
    *,
    requester: StorageRequester | None,
    authorized_route: Literal["/api/fits/jobs/{job_id}/cancel"] | None = None,
) -> StorageCancelRequest:
    """Build a cancellation with a fresh random request ID."""
    return StorageCancelRequest(
        schema_version=CANCEL_SCHEMA_VERSION,
        operation="cancel",
        request_id=secrets.token_hex(16),
        workspace=workspace,
        requester=requester,
        job_id=job_id,
        authorized_route=authorized_route,
    )


def exchange_cancel(
    channel: socket.socket,
    request: StorageCancelRequest,
    *,
    expected_service_uid: int,
    max_bytes: int,
    deadline: float,
    max_content_bytes: int | None = None,
) -> StudioJobRecord:
    """Run one cancellation over a connected, exclusively owned stream.

    Individual frames obey ``max_bytes``; the complete snapshot independently
    obeys ``max_content_bytes``, defaulting to event custody plus one frame.

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
        content_limit = view_content_limit(max_bytes, max_content_bytes)
        write_verified_frame(
            channel,
            encode_cancel_message(request),
            expected_uid=expected_service_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )
        reply = read_view_content(
            channel,
            expected_uid=expected_service_uid,
            frame_max_bytes=max_bytes,
            content_schema=CANCEL_SCHEMA_VERSION,
            request_id=request.request_id,
            max_content_bytes=content_limit,
            deadline=deadline,
        )
    response = decode_cancel_response(reply, request=request, max_bytes=content_limit)
    # An answered cancellation always carries its record.
    record = decode_job_snapshot(response.record or {})
    if record.job_id != request.job_id or record.workspace != request.workspace:
        raise ValueError("storage cancel record does not match the request")
    return record


__all__ = ["cancel_request", "exchange_cancel"]
