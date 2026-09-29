# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded finish artefact frames

"""Receive one declared artefact in deterministic, peer-verified frame chunks."""

import socket
import hashlib

from sc_neurocore.studio.platform.storage_peer import read_verified_frame


class FinishChunkMismatch(ValueError):
    """An otherwise framed chunk does not match its declared artefact position."""


def receive_finish_artifact(
    channel: socket.socket,
    *,
    size_bytes: int,
    sha256: str,
    expected_api_uid: int,
    frame_max_bytes: int,
    deadline: float,
) -> bytes:
    """Read exact full-size chunks and a final remainder under one deadline.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned API connection, with identity checked per frame.
    size_bytes : int
        Declared size already admitted under the aggregate artefact budget.
    sha256 : str
        Manifest digest to verify over the complete received bytes.
    expected_api_uid : int
        Operator-configured API identity.
    frame_max_bytes : int
        Positive maximum frame payload; every nonfinal chunk has this size.
    deadline : float
        Absolute monotonic deadline shared by the complete finish operation.

    Returns
    -------
    bytes
        Complete content for independent digest verification before sealing.
        An empty artefact consumes no frames.

    Raises
    ------
    ValueError
        Limits or chunk sizes are invalid.
    PermissionError, TimeoutError, EOFError, OSError
        Verified transfer cannot finish. No content is sealed here.
    """
    if type(size_bytes) is not int or size_bytes < 0:
        raise ValueError("invalid finish artefact size")
    if type(frame_max_bytes) is not int or not 0 < frame_max_bytes <= 0xFFFFFFFF:
        raise ValueError("invalid finish frame limit")
    chunks: list[bytes] = []
    remaining = size_bytes
    while remaining:
        chunk = read_verified_frame(
            channel,
            expected_uid=expected_api_uid,
            max_bytes=frame_max_bytes,
            deadline=deadline,
        )
        if len(chunk) != min(frame_max_bytes, remaining):
            raise FinishChunkMismatch("finish artefact chunk has the wrong size")
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise FinishChunkMismatch("finish artefact digest does not match")
    return payload
