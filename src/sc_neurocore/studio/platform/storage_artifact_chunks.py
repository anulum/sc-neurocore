# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded sealed artefact reception

"""Receive declared storage content within an independent API byte budget."""

import hashlib
import socket

from sc_neurocore.studio.platform.jobs_models import StudioJobArtifactRefused
from sc_neurocore.studio.platform.storage_finish_protocol import FinishArtifact
from sc_neurocore.studio.platform.storage_peer import read_verified_frame


def receive_artifact_chunks(
    channel: socket.socket,
    artifact: FinishArtifact,
    *,
    expected_service_uid: int,
    frame_max_bytes: int,
    max_artifact_bytes: int,
    deadline: float,
) -> bytes:
    """Read and verify one declaration under a locally trusted content ceiling.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned storage connection, verified at every frame.
    artifact : FinishArtifact
        Strict declaration from the correlated authority response.
    expected_service_uid : int
        Configured storage UID.
    frame_max_bytes, max_artifact_bytes : int
        Independent frame and complete-content limits configured at the API.
    deadline : float
        Absolute deadline shared with the request and declaration transfer.

    Returns
    -------
    bytes
        Exact content after full size and digest verification.

    Raises
    ------
    ValueError
        A local budget is invalid.
    StudioJobArtifactUnavailable
        Declared content exceeds the budget, chunks have incorrect boundaries,
        or the full content digest differs.
    PermissionError, TimeoutError, EOFError, OSError
        Peer verification or transfer fails.
    """
    if type(frame_max_bytes) is not int or not 0 < frame_max_bytes <= 0xFFFFFFFF:
        raise ValueError("invalid artifact frame limit")
    if type(max_artifact_bytes) is not int or max_artifact_bytes <= 0:
        raise ValueError("invalid artifact content limit")
    if artifact.size_bytes > max_artifact_bytes:
        raise StudioJobArtifactRefused("Studio job artifact exceeds the configured byte limit.")
    chunks: list[bytes] = []
    remaining = artifact.size_bytes
    while remaining:
        chunk = read_verified_frame(
            channel,
            expected_uid=expected_service_uid,
            max_bytes=frame_max_bytes,
            deadline=deadline,
        )
        if len(chunk) != min(frame_max_bytes, remaining):
            raise StudioJobArtifactRefused("Studio job artifact chunk size differs.")
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != artifact.sha256:
        raise StudioJobArtifactRefused("Studio job artifact integrity check failed.")
    return payload
