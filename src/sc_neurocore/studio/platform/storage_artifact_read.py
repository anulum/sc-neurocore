# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority sealed artefact reads

"""Serve one sealed artefact to the verified API after the route's policy.

Only an artefact the job's terminal record declares is served. Its bytes are
read from the authority's own sealed copy through held directory descriptors
that never follow a link, from a regular file this service owns, and must
match the declared size and SHA-256; anything else is ``unavailable``, never
a partial or substituted payload.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import socket
import stat

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_models import StudioJobArtifact
from sc_neurocore.studio.platform.jobs_paths import (
    _find_artifact,
    _normalize_artifact_lookup_path,
)
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.policy_routes import build_default_studio_route_policy_registry
from sc_neurocore.studio.platform.storage_artifact_protocol import (
    ARTIFACT_ROUTE_METHODS,
    ARTIFACT_SCHEMA_VERSION,
    StorageArtifactRequest,
    StorageArtifactResponse,
    decode_artifact_request,
    encode_artifact_message,
)
from sc_neurocore.studio.platform.storage_finish_protocol import FinishArtifact
from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_supervisor_identity,
    write_verified_frame,
)

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


def read_sealed_artifact(root: Path, job_id: str, artifact: StudioJobArtifact) -> bytes | None:
    """Return the sealed bytes of ``artifact``, or ``None`` when they cannot be trusted.

    Parameters
    ----------
    root : Path
        The authority root holding ``<job_id>/<relative path>``.
    job_id : str
        Job whose sealed directory is read.
    artifact : StudioJobArtifact
        The declaration from the job's terminal record.
    """
    *directories, name = (job_id, *artifact.relative_path.split("/"))
    held: list[int] = []
    try:
        held.append(os.open(root, _DIRECTORY))
        for directory in directories:
            held.append(os.open(directory, _DIRECTORY, dir_fd=held[-1]))
        handle = os.open(name, _FILE, dir_fd=held[-1])
        try:
            metadata = os.fstat(handle)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_size != artifact.size_bytes
            ):
                return None
            payload = os.read(handle, artifact.size_bytes + 1) if artifact.size_bytes else b""
        finally:
            os.close(handle)
    except OSError:
        return None
    finally:
        for descriptor in reversed(held):
            os.close(descriptor)
    if (
        len(payload) != artifact.size_bytes
        or hashlib.sha256(payload).hexdigest() != artifact.sha256
    ):
        return None
    return payload


def _answer(
    ledger: StudioJobLedger, request: StorageArtifactRequest, *, allowed: bool, max_bytes: int
) -> tuple[StorageArtifactResponse, bytes]:
    status = "forbidden"
    declared: FinishArtifact | None = None
    payload = b""
    if allowed:
        try:
            record = ledger.record(request.job_id, workspace=request.workspace)
            artifact = _find_artifact(
                record.artifacts, _normalize_artifact_lookup_path(request.relative_path)
            )
        except KeyError:
            status = "not_found"
        else:
            sealed = read_sealed_artifact(ledger.path.parent, request.job_id, artifact)
            if sealed is None or len(sealed) > max_bytes:
                status = "unavailable"
            else:
                status, payload = "ok", sealed
                declared = FinishArtifact(
                    relative_path=artifact.relative_path,
                    size_bytes=artifact.size_bytes,
                    sha256=artifact.sha256,
                )
    response = StorageArtifactResponse.model_validate(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "operation": "artifact",
            "request_id": request.request_id,
            "status": status,
            "artifact": None if declared is None else declared.model_dump(),
        },
        strict=True,
    )
    return response, payload


def serve_artifact_read(
    channel: socket.socket,
    *,
    ledger: StudioJobLedger,
    gateway: PolicyGateway,
    workspace: str,
    expected_api_uid: int,
    max_bytes: int,
    deadline: float,
    initial_frame: bytes | None = None,
) -> None:
    """Serve one peer-verified sealed artefact read.

    Parameters
    ----------
    channel : socket.socket
        Connected Unix stream, closed by this handler on every outcome.
    ledger : StudioJobLedger
        Existing authority; its root holds the sealed copies.
    gateway : PolicyGateway
        Existing authorisation owner with its audit sink.
    workspace : str
        Nonempty server-bound workspace.
    expected_api_uid : int
        Configured API identity.
    max_bytes : int
        Frame ceiling for the request, the answer and the artefact bytes.
    deadline : float
        Absolute monotonic wire deadline.
    initial_frame : bytes or None
        First frame already read by the owning listener, if any.

    Raises
    ------
    ValueError
        Configuration, request or workspace is invalid; nothing is answered.
    PermissionError
        The peer is not the configured API.
    TimeoutError, EOFError, OSError
        Wire transfer fails; reading again is safe.
    AuditSinkError
        The policy audit cannot persist its decision; nothing is read.
    """
    with channel:
        if not isinstance(workspace, str) or not workspace:
            raise ValueError("storage workspace must be nonempty")
        require_storage_supervisor_identity(channel, expected_uid=expected_api_uid)
        frame = initial_frame
        if frame is None:
            frame = read_verified_frame(
                channel, expected_uid=expected_api_uid, max_bytes=max_bytes, deadline=deadline
            )
        request = decode_artifact_request(frame, max_bytes=max_bytes)
        if request.workspace != workspace:
            raise ValueError("storage request does not match configured workspace")
        requester = request.requester
        principal = (
            None
            if requester is None
            else Principal(requester.principal_id, frozenset(requester.roles))
        )
        decision = gateway.authorize(
            build_default_studio_route_policy_registry().policy_for(
                ARTIFACT_ROUTE_METHODS[request.route], request.route
            ),
            principal=principal,
            route=request.route,
            request_id=request.request_id,
        )
        response, payload = _answer(ledger, request, allowed=decision.allowed, max_bytes=max_bytes)
        for frame_bytes in (encode_artifact_message(response), payload):
            if frame_bytes:
                write_verified_frame(
                    channel,
                    frame_bytes,
                    expected_uid=expected_api_uid,
                    max_bytes=max_bytes,
                    deadline=deadline,
                )


__all__ = ["read_sealed_artifact", "serve_artifact_read"]
