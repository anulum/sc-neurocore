# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority cancellation

"""Record a cancellation request at the authority, as the embedded manager does.

A live job moves to ``cancelling``; the API generation that supervises it
learns this from its next heartbeat and stops the worker. A job that already
stopped or is already cancelling is returned unchanged. When the job stops
between the read and the write, the ledger refuses the transition and the
record it reached is returned; a job the ledger will not cancel (``unknown``)
is returned as it is, and the API decides as the embedded manager would.
"""

from __future__ import annotations

import json
import socket

from pydantic import JsonValue

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import TERMINAL_STATUSES
from sc_neurocore.studio.platform.jobs_ledger_writes import transition_job
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.policy_routes import build_default_studio_route_policy_registry
from sc_neurocore.studio.platform.storage_cancel_protocol import (
    CANCEL_ROUTE,
    CANCEL_SCHEMA_VERSION,
    StorageCancelRequest,
    StorageCancelResponse,
    decode_cancel_request,
    encode_cancel_message,
)
from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_supervisor_identity,
    write_verified_frame,
)


def apply_cancel(
    ledger: StudioJobLedger, request: StorageCancelRequest, *, allowed: bool
) -> StorageCancelResponse:
    """Apply one decoded request whose workspace matched the service."""
    status: str = "forbidden"
    record: dict[str, JsonValue] | None = None
    if allowed:
        try:
            current = ledger.record(request.job_id, workspace=request.workspace)
        except KeyError:
            status = "not_found"
        else:
            if current.status not in TERMINAL_STATUSES and current.status != "cancelling":
                try:
                    current = transition_job(
                        ledger, request.job_id, "cancelling", reason="cancellation requested"
                    )
                except StudioJobRejected:
                    current = ledger.record(request.job_id, workspace=request.workspace)
            status = "ok"
            record = json.loads(json.dumps(current.to_public_dict(), allow_nan=False))
    return StorageCancelResponse.model_validate(
        {
            "schema_version": CANCEL_SCHEMA_VERSION,
            "operation": "cancel",
            "request_id": request.request_id,
            "job_id": request.job_id,
            "status": status,
            "record": record,
        },
        strict=True,
    )


def serve_cancel(
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
    """Serve one peer-verified cancellation after the stop route's policy.

    Parameters
    ----------
    channel : socket.socket
        Connected Unix stream, closed by this handler on every outcome.
    ledger : StudioJobLedger
        Existing authority.
    gateway : PolicyGateway
        Existing authorisation owner with its audit sink.
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
        Configuration, request or workspace is invalid; nothing is answered.
    PermissionError
        The peer is not the configured API.
    TimeoutError, EOFError, OSError
        Wire transfer fails; repeating the request is safe.
    AuditSinkError
        The policy audit cannot persist its decision; nothing is changed.
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
        request = decode_cancel_request(frame, max_bytes=max_bytes)
        if request.workspace != workspace:
            raise ValueError("storage request does not match configured workspace")
        requester = request.requester
        principal = (
            None
            if requester is None
            else Principal(requester.principal_id, frozenset(requester.roles))
        )
        method, route = CANCEL_ROUTE
        decision = gateway.authorize(
            build_default_studio_route_policy_registry().policy_for(method, route),
            principal=principal,
            route=route,
            request_id=request.request_id,
        )
        response = apply_cancel(ledger, request, allowed=decision.allowed)
        write_verified_frame(
            channel,
            encode_cancel_message(response),
            expected_uid=expected_api_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )


__all__ = ["apply_cancel", "serve_cancel"]
