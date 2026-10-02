# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API client for storage purge

"""Ask the storage authority to purge one terminal job."""

from __future__ import annotations

import secrets
import socket

from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot
from sc_neurocore.studio.platform.storage_peer import write_verified_frame
from sc_neurocore.studio.platform.storage_purge_protocol import (
    PURGE_SCHEMA_VERSION,
    StoragePurgeRequest,
    decode_purge_response,
    encode_purge_message,
)
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester
from sc_neurocore.studio.platform.storage_snapshot import storage_job_record
from sc_neurocore.studio.platform.storage_view_content import read_view_content, view_content_limit


def purge_request(
    workspace: str, job_id: str, *, requester: StorageRequester | None
) -> StoragePurgeRequest:
    """Build a purge with a fresh random request ID."""
    return StoragePurgeRequest(
        schema_version=PURGE_SCHEMA_VERSION,
        operation="purge",
        request_id=secrets.token_hex(16),
        workspace=workspace,
        requester=requester,
        job_id=job_id,
    )


def exchange_purge(
    channel: socket.socket,
    request: StoragePurgeRequest,
    *,
    expected_service_uid: int,
    max_bytes: int,
    deadline: float,
    max_content_bytes: int | None = None,
) -> StudioJobRecord:
    """Run one purge over a connected, exclusively owned stream.

    Frames obey ``max_bytes``; the full pre-purge snapshot independently obeys
    ``max_content_bytes``, defaulting to event custody plus one frame.

    Returns
    -------
    StudioJobRecord
        The record as it was before the purge.

    Raises
    ------
    PermissionError
        The peer is not the configured storage identity, or policy denied.
    KeyError
        The job is not in the workspace.
    StudioJobRejected
        The ledger refused the purge.
    ValueError
        The reply is malformed or names another job.
    TimeoutError, EOFError, OSError
        The exchange failed; read the record to learn whether it was purged.
    """
    with channel:
        content_limit = view_content_limit(max_bytes, max_content_bytes)
        write_verified_frame(
            channel,
            encode_purge_message(request),
            expected_uid=expected_service_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )
        reply = read_view_content(
            channel,
            expected_uid=expected_service_uid,
            frame_max_bytes=max_bytes,
            content_schema=PURGE_SCHEMA_VERSION,
            request_id=request.request_id,
            max_content_bytes=content_limit,
            deadline=deadline,
        )
    response = decode_purge_response(reply, request=request, max_bytes=content_limit)
    # A purge always carries the purged record.
    record = decode_job_snapshot(response.record or {})
    if record.job_id != request.job_id or record.workspace != request.workspace:
        raise ValueError("storage purge record does not match the request")
    return storage_job_record(record)


__all__ = ["exchange_purge", "purge_request"]
