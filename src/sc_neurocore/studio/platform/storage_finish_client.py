# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API client for delegated job finish

"""Read a stopped worker's spool as untrusted input and finish its job.

The worker's own result file and every artefact it declares are read through
a held job directory, never following symbolic links, accepting only regular
files whose identity and size stay unchanged while they are read. Digests are
computed here, never copied from the worker. A timeout or disconnect after
``ready`` is ambiguous; repeating the identical request is safe because the
authority answers ``already_sealed`` for an identical finished job.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import stat
from collections.abc import Sequence

from pydantic import JsonValue

from sc_neurocore.studio.platform.storage_finish_protocol import (
    FINISH_SCHEMA_VERSION,
    FinishArtifact,
    FinishOutcome,
    StorageFinishRequest,
    StorageFinishResponse,
    decode_finish_response,
    encode_finish_message,
    validate_artifact_budget,
)
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame

RESULT_NAME = ".studio_process_result.json"
# Non-blocking so that a named pipe planted by the worker cannot stall the API.
_READ = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


def _read_stable(name: str, parent: int, *, limit: int) -> bytes:
    """Read one regular file relative to ``parent`` that does not change meanwhile.

    Raises
    ------
    ValueError
        The entry is not a regular file, exceeds ``limit`` or changed while read.
    OSError
        It cannot be opened without following links.
    """
    descriptor = os.open(name, _READ, dir_fd=parent)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("spool entry is not a bounded regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (before.st_ino, before.st_size, before.st_mtime_ns)
    if (
        identity != (after.st_ino, after.st_size, after.st_mtime_ns)
        or len(payload) != after.st_size
    ):
        raise ValueError("spool entry changed while it was read")
    return payload


def _open_parent(relative_path: str, job: int) -> tuple[int, str]:
    """Walk to an artefact's directory without following links."""
    *directories, name = relative_path.split("/")
    parent = os.dup(job)
    try:
        for directory in directories:
            child = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | _READ, dir_fd=parent)
            os.close(parent)
            parent = child
    except BaseException:
        os.close(parent)
        raise
    return parent, name


def _exit_message(exit_status: int | None) -> str:
    if exit_status is None:
        return "Studio process worker has no recorded exit status."
    return f"Studio process worker exited with {exit_status}."


def spool_finish_request(
    job_directory: int,
    *,
    workspace: str,
    job_id: str,
    outcome: FinishOutcome | None,
    exit_status: int | None,
    frame_max_bytes: int,
    max_artifact_bytes: int,
    max_artifact_entries: int,
    error: str | None = None,
    worker_reaped: bool = True,
) -> tuple[StorageFinishRequest, tuple[bytes, ...]]:
    """Build a finish request and its artefact bytes from a stopped worker's spool.

    Parameters
    ----------
    job_directory : int
        Held descriptor of ``<spool>/<job>/<generation>/<job>``.
    workspace, job_id : str
        Configured workspace and the admitted job.
    outcome : FinishOutcome or None
        The API's own verdict (``cancelled``/``timed_out``/``failed``) or
        ``None`` to use the worker's report: ``completed`` only when the worker
        reported completion and exited with status 0, as embedded.
    exit_status : int or None
        Leader exit status reported by the launcher, if it ran.
    frame_max_bytes : int
        Ceiling for the result file and for each artefact.
    max_artifact_bytes, max_artifact_entries : int
        Aggregate budgets the authority enforces; a worker declaring more is
        refused before any artefact is read into memory.
    error : str or None
        The API's own error for an unsuccessful outcome; otherwise the worker's
        error or the embedded supervisor's wording is used.
    worker_reaped : bool
        Whether the launcher confirmed that every process of the generation
        ended.

    Returns
    -------
    tuple
        The validated request and the artefact bytes in manifest order.

    Raises
    ------
    ValueError
        The worker result or an artefact is missing its contract: absent or
        non-regular files, changed bytes, a digest or size different from the
        declaration, or an invalid manifest.
    """
    try:
        raw = json.loads(_read_stable(RESULT_NAME, job_directory, limit=frame_max_bytes))
    except FileNotFoundError:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("worker result is not an object")
    declared = raw.get("artifacts", [])
    if not isinstance(declared, list):
        raise ValueError("worker artefact manifest is not a list")
    if len(declared) > max_artifact_entries:
        raise ValueError("worker artefact manifest exceeds entry limit")
    artifacts = [FinishArtifact.model_validate(entry, strict=True) for entry in declared]
    validate_artifact_budget(
        artifacts,
        frame_max_bytes=frame_max_bytes,
        max_artifact_bytes=max_artifact_bytes,
        max_artifact_entries=max_artifact_entries,
    )
    payloads: list[bytes] = []
    for artifact in artifacts:
        parent, name = _open_parent(artifact.relative_path, job_directory)
        try:
            payload = _read_stable(name, parent, limit=artifact.size_bytes)
        finally:
            os.close(parent)
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != artifact.size_bytes or digest != artifact.sha256:
            raise ValueError("artefact bytes differ from the worker declaration")
        payloads.append(payload)
    reported = raw.get("status") == "completed"
    verdict: FinishOutcome = outcome or ("completed" if reported and exit_status == 0 else "failed")
    worker_error = raw.get("error")
    message = error
    if message is None and verdict == "failed":
        valid = isinstance(worker_error, str) and 0 < len(worker_error) <= 1024
        message = worker_error if valid else _exit_message(exit_status)
    elif message is None and verdict == "timed_out":
        message = "Studio job exceeded its timeout."
    result = raw.get("result")
    request = StorageFinishRequest(
        schema_version=FINISH_SCHEMA_VERSION,
        operation="finish",
        request_id=secrets.token_hex(16),
        workspace=workspace,
        job_id=job_id,
        outcome=verdict,
        result=_result(result) if verdict == "completed" else None,
        error=message,
        artifacts=tuple(artifacts),
        worker_reaped=worker_reaped,
    )
    return request, tuple(payloads)


def _result(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in json.loads(json.dumps(value)).items()}


def exchange_finish(
    channel: socket.socket,
    request: StorageFinishRequest,
    payloads: Sequence[bytes],
    *,
    expected_service_uid: int,
    max_bytes: int,
    deadline: float,
) -> StorageFinishResponse:
    """Run one finish exchange over a connected, exclusively owned stream.

    Parameters
    ----------
    channel : socket.socket
        Connected Unix stream to the storage authority, closed on every outcome.
    request : StorageFinishRequest
        Request to send.
    payloads : sequence of bytes
        Artefact bytes in manifest order.
    expected_service_uid : int
        Configured storage identity, checked before each frame.
    max_bytes : int
        Frame ceiling.
    deadline : float
        Absolute monotonic deadline for the whole exchange.

    Returns
    -------
    StorageFinishResponse
        The final answer.

    Raises
    ------
    ValueError
        ``payloads`` do not match the manifest, or a reply is malformed.
    PermissionError
        The peer is not the configured storage identity.
    TimeoutError, EOFError, OSError
        The exchange failed; whether the authority sealed is unknown.
    """
    if len(payloads) != len(request.artifacts):
        raise ValueError("artefact bytes do not match the manifest")

    def send(payload: bytes) -> None:
        write_verified_frame(
            channel,
            payload,
            expected_uid=expected_service_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )

    def receive() -> StorageFinishResponse:
        reply = read_verified_frame(
            channel, expected_uid=expected_service_uid, max_bytes=max_bytes, deadline=deadline
        )
        return decode_finish_response(reply, request=request, max_bytes=max_bytes)

    with channel:
        send(encode_finish_message(request))
        response = receive()
        if response.reply != "ready":
            return response
        for payload in payloads:
            if payload:
                send(payload)
        return receive()
