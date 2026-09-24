# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job supervision authority

"""Apply start and heartbeat for the verified API generation that owns a job.

The storage authority derives the acting supervisor from the connection's
pidfd-verified API peer and compares it with the job's recorded lease owner
and capacity reservation, which delegated admission set to that generation.
``start`` marks the job running under that delegated lease and registers the
observed worker generation through the existing custody owner; a job already
cancelling keeps its cancellation and receives no worker. ``heartbeat``
renews only the delegated owner's live lease. No payload, artifact, path or
supervisor value is read from the wire.

Ownership is checked before acting and again by the ledger writers under
their own write lock. A job that ends in between is reported ``not_live``; a
worker that exits in between is reported ``worker_unverified``. The recorded
lease owner of a live job never changes, so a writer refusing the verified
owner, or a job purged in between, is not an outcome: the exception closes the
exchange without a reply and a retried request observes the settled state.
"""

from __future__ import annotations

import socket

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import TERMINAL_STATUSES
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_is_alive
from sc_neurocore.studio.platform.jobs_ledger_writes import heartbeat_job, transition_job
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.jobs_worker_custody import register_worker
from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_supervisor_identity,
    write_verified_frame,
)
from sc_neurocore.studio.platform.storage_supervision_protocol import (
    SUPERVISION_SCHEMA_VERSION,
    StorageSupervisionResponse,
    SupervisionHeartbeatRequest,
    SupervisionStartRequest,
    SupervisionOutcome,
    SupervisionReason,
    decode_supervision_request,
    encode_supervision_message,
)


def _owner_status(
    ledger: StudioJobLedger, job_id: str, workspace: str
) -> tuple[str, str, str | None, str | None] | None:
    """Return status, lease owner, reservation owner and registered worker."""
    with ledger.transaction() as connection:
        row = connection.execute(
            "SELECT j.status, j.lease_owner, r.supervisor AS reserved, w.worker_identity "
            "FROM jobs j LEFT JOIN admission_reservations r ON r.job_id = j.job_id "
            "LEFT JOIN job_workers w ON w.job_id = j.job_id "
            "WHERE j.job_id = ? AND j.workspace = ?",
            (job_id, workspace),
        ).fetchone()
    if row is None:
        return None
    return str(row["status"]), str(row["lease_owner"]), row["reserved"], row["worker_identity"]


def apply_supervision(
    ledger: StudioJobLedger,
    request: SupervisionStartRequest | SupervisionHeartbeatRequest,
    *,
    supervisor: str,
    workspace: str,
) -> StorageSupervisionResponse:
    """Apply one decoded request on behalf of the verified API generation.

    Parameters
    ----------
    ledger : StudioJobLedger
        Existing storage authority.
    request : SupervisionStartRequest or SupervisionHeartbeatRequest
        Request whose workspace was already matched to ``workspace``.
    supervisor : str
        ``host:pid:token`` of the pidfd-verified API peer.
    workspace : str
        Server-configured workspace; jobs elsewhere are reported not found.

    Returns
    -------
    StorageSupervisionResponse
        Outcome correlated with ``request``.
    """
    outcome: SupervisionOutcome = "refused"
    reason: SupervisionReason | None = None
    observed = _owner_status(ledger, request.job_id, workspace)
    if observed is None:
        reason = "not_found"
    else:
        status, lease_owner, reserved, registered = observed
        if status in TERMINAL_STATUSES or status == "unknown":
            reason = "not_live"
        elif lease_owner != supervisor or reserved != supervisor:
            reason = "not_owner"
        elif isinstance(request, SupervisionHeartbeatRequest):
            if heartbeat_job(ledger, request.job_id, supervisor=supervisor):
                # A cancellation recorded after this observation is reported
                # by the next heartbeat.
                outcome = "cancelling" if status == "cancelling" else "renewed"
            else:
                reason = "not_live"
        elif registered is not None:
            if registered == request.worker:
                outcome = "started"
            else:
                reason = "worker_conflict"
        else:
            outcome, reason = _start(ledger, request, supervisor=supervisor)
    return StorageSupervisionResponse(
        schema_version=SUPERVISION_SCHEMA_VERSION,
        operation=request.operation,
        request_id=request.request_id,
        job_id=request.job_id,
        outcome=outcome,
        reason=reason,
    )


def _start(
    ledger: StudioJobLedger, request: SupervisionStartRequest, *, supervisor: str
) -> tuple[SupervisionOutcome, SupervisionReason | None]:
    worker = request.worker
    if supervisor_is_alive(worker) is not True:
        return "refused", "worker_unverified"
    try:
        record = transition_job(
            ledger,
            request.job_id,
            "running",
            started_at_utc=ledger.timestamp(),
            supervisor=supervisor,
        )
    except StudioJobRejected:
        # The job ended after ownership was checked.
        return "refused", "not_live"
    if record.status == "cancelling":
        return "cancelling", None
    try:
        register_worker(ledger, request.job_id, supervisor, worker, int(worker.split(":")[1]))
    except (ValueError, OSError):
        return "refused", "worker_unverified"
    return "started", None


def serve_supervision(
    channel: socket.socket,
    *,
    ledger: StudioJobLedger,
    workspace: str,
    expected_api_uid: int,
    max_bytes: int,
    deadline: float,
    initial_frame: bytes | None = None,
) -> None:
    """Serve one peer-verified supervision request and write its outcome.

    Parameters
    ----------
    channel : socket.socket
        Connected Unix stream, closed by this handler on every outcome.
    ledger : StudioJobLedger
        Existing authority.
    workspace : str
        Nonempty server-bound workspace.
    expected_api_uid : int
        Configured API identity.
    max_bytes : int
        Frame ceiling for request and response.
    deadline : float
        Absolute monotonic wire deadline.
    initial_frame : bytes or None
        First frame already read by the owning listener, if any.

    Raises
    ------
    ValueError
        Configuration, request or workspace is invalid; nothing is written.
    PermissionError
        The peer is not the configured API or its generation cannot be proved.
    TimeoutError, EOFError, OSError
        Wire transfer fails.
    """
    with channel:
        if not isinstance(workspace, str) or not workspace:
            raise ValueError("storage workspace must be nonempty")
        supervisor = require_storage_supervisor_identity(channel, expected_uid=expected_api_uid)
        metadata = initial_frame
        if metadata is None:
            metadata = read_verified_frame(
                channel, expected_uid=expected_api_uid, max_bytes=max_bytes, deadline=deadline
            )
        request = decode_supervision_request(metadata, max_bytes=max_bytes)
        if request.workspace != workspace:
            raise ValueError("storage request does not match configured workspace")
        response = apply_supervision(ledger, request, supervisor=supervisor, workspace=workspace)
        write_verified_frame(
            channel,
            encode_supervision_message(response),
            expected_uid=expected_api_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )
