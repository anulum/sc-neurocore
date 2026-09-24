# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — one-shot storage record read

"""Read one record through existing policy and ledger authorities.

This connected-socket handler is not a production listener or isolated profile.
Only the configured trusted API peer may delegate its authenticated principal;
OS UID acceptance alone cannot authenticate a browser user.
"""

import json
import socket

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.policy_routes import build_default_studio_route_policy_registry
from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_supervisor_identity,
    write_verified_frame,
)
from sc_neurocore.studio.platform.storage_record_protocol import decode_record_request

_ROUTE = "/api/studio/jobs/{job_id}"


def serve_record_read(
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
    """Serve one peer-verified read with policy evaluation before ledger lookup.

    Parameters
    ----------
    channel : socket.socket
        Connected Unix stream, owned and closed by this handler on every outcome.
    ledger : StudioJobLedger
        Existing authority; no client-selected root, SQL or embedded fallback.
    gateway : PolicyGateway
        Existing authorization owner with its configured audit sink.
    workspace : str
        Nonempty server-bound scope; a request cannot select another workspace.
    expected_api_uid : int
        Trusted API OS identity, distinct from workers in a qualified deployment.
    max_bytes : int
        Explicit frame byte limit for both request and complete response.
    deadline : float
        Absolute monotonic wire deadline. Audit/SQLite execution has separate
        bounds; this is not a service-wide scheduling deadline.
    initial_frame : bytes or None
        Optional first frame already read through the same peer-verified channel
        by the owning listener. Direct callers leave this unset.

    Raises
    ------
    ValueError
        Configuration or wire request is invalid, or response exceeds its limit.
    PermissionError
        The connection peer is not the configured trusted API.
    TimeoutError
        A wire transfer exceeds the deadline.
    EOFError
        Peer disconnects during a frame.
    OSError
        Socket transfer fails.
    AuditSinkError
        Existing policy audit cannot persist its decision; no record is sent.

    Notes
    -----
    Denied/missing reads return path-free outcomes, not internal exception text.
    Admin access remains cross-owner within the server-bound workspace. No job
    state, reservation, lease or transition is changed by this operation.
    """
    with channel:
        if not isinstance(workspace, str) or not workspace:
            raise ValueError("storage workspace must be nonempty")
        require_storage_supervisor_identity(channel, expected_uid=expected_api_uid)
        metadata = initial_frame
        if metadata is None:
            metadata = read_verified_frame(
                channel, expected_uid=expected_api_uid, max_bytes=max_bytes, deadline=deadline
            )
        if not isinstance(metadata, bytes) or not 0 < len(metadata) <= max_bytes:
            raise ValueError("storage record frame exceeds byte limit")
        request = decode_record_request(metadata)
        if request.workspace != workspace:
            raise ValueError("storage request does not match configured workspace")
        principal = (
            None
            if request.requester is None
            else Principal(request.requester.principal_id, frozenset(request.requester.roles))
        )
        policy = build_default_studio_route_policy_registry().policy_for("GET", _ROUTE)
        decision = gateway.authorize(
            policy, principal=principal, route=_ROUTE, request_id=request.request_id
        )
        response: dict[str, object] = {
            "schema_version": "studio.storage.record.v2",
            "request_id": request.request_id,
            "status": "forbidden",
            "record": None,
        }
        if decision.allowed:
            try:
                record = ledger.record(request.job_id, workspace=workspace)
            except KeyError:
                response["status"] = "not_found"
            else:
                response["status"] = "ok"
                response["record"] = record.to_public_dict()
        write_verified_frame(
            channel,
            json.dumps(response, allow_nan=False, sort_keys=True).encode("utf-8"),
            expected_uid=expected_api_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )
