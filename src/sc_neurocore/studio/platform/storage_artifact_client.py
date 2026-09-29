# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API client for sealed artefact reads

"""Read one sealed artefact from the authority and verify it end to end.

The API checks the received bytes against the declared size and SHA-256
itself, so a damaged transfer never reaches a caller. Reads change nothing;
a lost reply is read again on a new connection.
"""

from __future__ import annotations

import secrets
import socket

from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifact,
    StudioJobArtifactPayload,
    StudioJobArtifactUnavailable,
)
from sc_neurocore.studio.platform.storage_artifact_protocol import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactRoute,
    StorageArtifactRequest,
    decode_artifact_response,
    encode_artifact_message,
)
from sc_neurocore.studio.platform.storage_artifact_chunks import receive_artifact_chunks
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester


def artifact_request(
    workspace: str,
    job_id: str,
    relative_path: str,
    *,
    route: ArtifactRoute,
    requester: StorageRequester | None,
) -> StorageArtifactRequest:
    """Build a sealed artefact read with a fresh random request ID."""
    return StorageArtifactRequest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        operation="artifact",
        request_id=secrets.token_hex(16),
        workspace=workspace,
        requester=requester,
        route=route,
        job_id=job_id,
        relative_path=relative_path,
    )


def exchange_artifact(
    channel: socket.socket,
    request: StorageArtifactRequest,
    *,
    expected_service_uid: int,
    max_bytes: int,
    deadline: float,
    max_artifact_bytes: int = 64 * 1024 * 1024,
) -> StudioJobArtifactPayload:
    """Read one sealed artefact over a connected, exclusively owned stream.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned connection, closed after this exchange.
    request : StorageArtifactRequest
        Route-authorised read with a fresh correlation identifier.
    expected_service_uid : int
        Configured storage identity, checked for each frame.
    max_bytes, max_artifact_bytes : int
        Independent frame and complete-content ceilings from trusted settings.
    deadline : float
        Absolute monotonic deadline covering the entire exchange.

    Returns
    -------
    StudioJobArtifactPayload
        Complete bytes after exact chunk, size and digest verification.

    Raises
    ------
    PermissionError
        The peer is not the configured storage identity, or policy denied.
    KeyError
        The job or its declared artefact is not in the workspace.
    StudioJobArtifactUnavailable
        The authority could not serve trusted bytes, or the received bytes
        differ from the declaration.
    ValueError
        A reply is malformed or answers another request or artefact.
    TimeoutError, EOFError, OSError
        The exchange failed; reading again is safe.
    """
    with channel:
        write_verified_frame(
            channel,
            encode_artifact_message(request),
            expected_uid=expected_service_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )
        reply = read_verified_frame(
            channel, expected_uid=expected_service_uid, max_bytes=max_bytes, deadline=deadline
        )
        declared = decode_artifact_response(reply, request=request, max_bytes=max_bytes).artifact
        if declared is None:
            raise StudioJobArtifactUnavailable("Studio job artifact declaration is absent.")
        payload = receive_artifact_chunks(
            channel,
            declared,
            expected_service_uid=expected_service_uid,
            frame_max_bytes=max_bytes,
            max_artifact_bytes=max_artifact_bytes,
            deadline=deadline,
        )
    artifact = StudioJobArtifact(
        relative_path=declared.relative_path, size_bytes=declared.size_bytes, sha256=declared.sha256
    )
    return StudioJobArtifactPayload(artifact=artifact, payload=payload)


__all__ = ["artifact_request", "exchange_artifact"]
