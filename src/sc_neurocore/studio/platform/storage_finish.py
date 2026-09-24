# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job finish authority

"""Seal a finished job's artefacts and commit its terminal record.

The storage authority acts for the pidfd-verified API generation that owns the
job's delegated lease and reservation. It answers ``ready`` only for a live
job whose registered worker is provably gone, or for a job that never started
a worker and ends unsuccessfully; every declared artefact must
then arrive with its exact size and SHA-256 before any byte is sealed. Sealing
precedes the terminal transition, and an identical retry of a finished job is
answered ``already_sealed``, so a lost reply never seals twice or seals
different bytes. The reservation is released with the terminal record, or
kept as unreaped when the API could not confirm that the worker's processes
ended.
"""

from __future__ import annotations

import hashlib
import socket
import sqlite3

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_rows import artifacts_from_json
from sc_neurocore.studio.platform.jobs_ledger_schema import TERMINAL_STATUSES
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_is_alive
from sc_neurocore.studio.platform.jobs_ledger_writes import transition_job
from sc_neurocore.studio.platform.jobs_models import StudioJobArtifact, StudioJobRejected
from sc_neurocore.studio.platform.storage_artifact_seal import SealedArtifactWriter
from sc_neurocore.studio.platform.storage_finish_protocol import (
    FINISH_SCHEMA_VERSION,
    FinishReason,
    FinishReply,
    StorageFinishRequest,
    StorageFinishResponse,
    decode_finish_request,
    encode_finish_message,
    validate_finish_manifest,
)
from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_supervisor_identity,
    write_verified_frame,
)

_Answer = tuple[FinishReply, FinishReason | None]


def _declared(request: StorageFinishRequest) -> tuple[StudioJobArtifact, ...]:
    return tuple(
        StudioJobArtifact(
            relative_path=artifact.relative_path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        )
        for artifact in request.artifacts
    )


def _observe(ledger: StudioJobLedger, request: StorageFinishRequest) -> sqlite3.Row | None:
    """Read status, ownership and registered worker in one snapshot."""
    with ledger.transaction() as connection:
        row: sqlite3.Row | None = connection.execute(
            "SELECT j.status, j.lease_owner, j.artifacts, r.supervisor AS reserved, "
            "w.worker_identity FROM jobs j "
            "LEFT JOIN admission_reservations r ON r.job_id = j.job_id "
            "LEFT JOIN job_workers w ON w.job_id = j.job_id "
            "WHERE j.job_id = ? AND j.workspace = ?",
            (request.job_id, request.workspace),
        ).fetchone()
    return row


def _settled(row: sqlite3.Row | None, request: StorageFinishRequest) -> _Answer:
    """Answer a request for a job that already has a terminal record.

    A job that keeps its reservation cannot be purged, so ``row`` is present;
    an absent row would never match and is answered as a conflict.
    """
    identical = (
        row is not None
        and str(row["status"]) == request.outcome
        and artifacts_from_json(str(row["artifacts"])) == _declared(request)
    )
    return ("already_sealed", None) if identical else ("refused", "conflict")


def decide_finish(
    ledger: StudioJobLedger, request: StorageFinishRequest, *, supervisor: str
) -> _Answer:
    """Decide whether the verified API generation may finish this job now.

    Parameters
    ----------
    ledger : StudioJobLedger
        Existing storage authority.
    request : StorageFinishRequest
        Decoded request whose workspace already matched the service.
    supervisor : str
        ``host:pid:token`` of the pidfd-verified API peer.

    Returns
    -------
    tuple
        ``("ready", None)`` to receive artefacts, or a final reply and reason.
    """
    row = _observe(ledger, request)
    if row is None:
        return "refused", "not_found"
    status = str(row["status"])
    if status in TERMINAL_STATUSES:
        return _settled(row, request)
    worker = row["worker_identity"]
    unstarted = status == "pending" and worker is None and request.outcome != "completed"
    if status not in ("running", "cancelling") and not unstarted:
        return "refused", "not_live"
    if row["lease_owner"] != supervisor or row["reserved"] != supervisor:
        return "refused", "not_owner"
    if worker is not None and supervisor_is_alive(str(worker)) is not False:
        return "refused", "worker_live"
    return "ready", None


def commit_finish(
    ledger: StudioJobLedger, request: StorageFinishRequest, *, supervisor: str
) -> _Answer:
    """Commit the terminal record for sealed artefacts and settle its capacity.

    Both writes are one transaction: a crash never leaves a terminal job that
    still holds its reservation, which an identical retry could not release.
    An unreaped worker keeps the reservation, marked ``unreaped``, as the
    embedded supervisor does.

    Parameters
    ----------
    ledger : StudioJobLedger
        Existing storage authority.
    request : StorageFinishRequest
        Request whose artefacts are already sealed.
    supervisor : str
        Delegated owner verified before the artefacts were received.

    Returns
    -------
    tuple
        ``sealed``, or the answer for a job that became terminal meanwhile.
    """
    try:
        with ledger.transaction() as connection:
            transition_job(
                ledger,
                request.job_id,
                request.outcome,
                supervisor=supervisor,
                finished_at_utc=ledger.timestamp(),
                error=request.error,
                result=request.result,
                artifacts=_declared(request),
                connection=connection,
            )
            connection.execute(
                "DELETE FROM admission_reservations "
                "WHERE job_id = ? AND supervisor = ? AND state != 'unreaped'"
                if request.worker_reaped
                else "UPDATE admission_reservations SET state = 'unreaped' "
                "WHERE job_id = ? AND supervisor = ?",
                (request.job_id, supervisor),
            )
    except StudioJobRejected:
        # The job reached another terminal state after the ownership check.
        return _settled(_observe(ledger, request), request)
    return "sealed", None


def _answer(
    channel: socket.socket,
    request: StorageFinishRequest,
    answer: _Answer,
    *,
    expected_api_uid: int,
    max_bytes: int,
    deadline: float,
) -> None:
    reply, reason = answer
    response = StorageFinishResponse(
        schema_version=FINISH_SCHEMA_VERSION,
        operation="finish",
        request_id=request.request_id,
        job_id=request.job_id,
        reply=reply,
        reason=reason,
    )
    write_verified_frame(
        channel,
        encode_finish_message(response),
        expected_uid=expected_api_uid,
        max_bytes=max_bytes,
        deadline=deadline,
    )


def serve_finish(
    channel: socket.socket,
    *,
    ledger: StudioJobLedger,
    workspace: str,
    expected_api_uid: int,
    frame_max_bytes: int,
    max_artifact_bytes: int,
    max_artifact_entries: int,
    deadline: float,
    initial_frame: bytes | None = None,
) -> None:
    """Serve one peer-verified finish exchange and write its final answer.

    Parameters
    ----------
    channel : socket.socket
        Connected Unix stream, closed by this handler on every outcome.
    ledger : StudioJobLedger
        Existing authority; artefacts are sealed under its root.
    workspace : str
        Nonempty server-bound workspace.
    expected_api_uid : int
        Configured API identity.
    frame_max_bytes : int
        Ceiling for every frame, including one artefact.
    max_artifact_bytes, max_artifact_entries : int
        Trusted aggregate artefact budgets.
    deadline : float
        Absolute monotonic deadline for the whole exchange.
    initial_frame : bytes or None
        First frame already read by the owning listener, if any.

    Raises
    ------
    ValueError
        Configuration, request, workspace or manifest is invalid; nothing is
        answered or sealed.
    PermissionError
        The peer is not the configured API or its generation cannot be proved.
    TimeoutError, EOFError, OSError
        Wire transfer fails; bytes sealed before the failure are only retained
        for an identical retry.
    """
    with channel:
        if not isinstance(workspace, str) or not workspace:
            raise ValueError("storage workspace must be nonempty")
        supervisor = require_storage_supervisor_identity(channel, expected_uid=expected_api_uid)
        frame = initial_frame
        if frame is None:
            frame = read_verified_frame(
                channel, expected_uid=expected_api_uid, max_bytes=frame_max_bytes, deadline=deadline
            )
        request = decode_finish_request(frame, max_bytes=frame_max_bytes)
        if request.workspace != workspace:
            raise ValueError("storage request does not match configured workspace")
        validate_finish_manifest(
            request,
            frame_max_bytes=frame_max_bytes,
            max_artifact_bytes=max_artifact_bytes,
            max_artifact_entries=max_artifact_entries,
        )
        wire = {"expected_api_uid": expected_api_uid, "max_bytes": frame_max_bytes}
        answer = decide_finish(ledger, request, supervisor=supervisor)
        _answer(channel, request, answer, deadline=deadline, **wire)
        if answer[0] != "ready":
            return
        received: list[bytes] = []
        for artifact in request.artifacts:
            payload = b""
            if artifact.size_bytes:
                payload = read_verified_frame(
                    channel,
                    expected_uid=expected_api_uid,
                    max_bytes=frame_max_bytes,
                    deadline=deadline,
                )
            if (
                len(payload) != artifact.size_bytes
                or hashlib.sha256(payload).hexdigest() != artifact.sha256
            ):
                _answer(channel, request, ("refused", "bytes"), deadline=deadline, **wire)
                return
            received.append(payload)
        try:
            with SealedArtifactWriter(ledger.path.parent, request.job_id) as writer:
                for artifact, payload in zip(request.artifacts, received, strict=True):
                    writer.seal(artifact.relative_path, payload, sha256=artifact.sha256)
        except FileExistsError:
            _answer(channel, request, ("refused", "conflict"), deadline=deadline, **wire)
            return
        answer = commit_finish(ledger, request, supervisor=supervisor)
        _answer(channel, request, answer, deadline=deadline, **wire)
