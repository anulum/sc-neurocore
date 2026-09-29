# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority read-only views

"""Serve bounded read-only views of the authority to the verified API.

Each view is authorised by the existing policy of the HTTP route it serves,
for the requester the API delegated, before any ledger read. Record pages
follow creation order (creation time, then insertion order), as the embedded
manager lists them; the cursor is the last job returned and must still exist
in the workspace. Nothing is written: no transition, lease or reservation
changes, and no recovery runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import socket

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.policy_routes import build_default_studio_route_policy_registry
from sc_neurocore.studio.platform.storage_query_pages import fit_page, read_record_page
from sc_neurocore.studio.platform.storage_view_content import send_view_content, view_content_limit
from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_supervisor_identity,
)
from sc_neurocore.studio.platform.storage_query_protocol import (
    QUERY_ROUTES,
    QUERY_SCHEMA_VERSION,
    QueryStatus,
    StorageQueryRequest,
    StorageQueryResponse,
    decode_query_request,
    encode_query_message,
)
from pydantic import JsonValue

_Page = tuple[list[dict[str, JsonValue]], bool]


def _plain(item: Mapping[str, object]) -> dict[str, JsonValue]:
    """Return a public dictionary as JSON values, refusing non-JSON content."""
    decoded = json.loads(json.dumps(item, allow_nan=False))
    return {str(key): value for key, value in decoded.items()}


def _purges(ledger: StudioJobLedger, request: StorageQueryRequest) -> _Page:
    snapshot = ledger.purge_snapshot(limit=request.limit, after=request.after)
    items = [_plain(purge.to_public_dict()) for purge in snapshot.purges]
    return items, snapshot.next_after is not None


def _summary(
    ledger: StudioJobLedger, admission: SharedJobAdmission, workspace: str
) -> dict[str, JsonValue]:
    """Count the workspace's jobs by status and model; list its unreaped jobs."""
    with ledger.transaction() as connection:
        statuses = connection.execute(
            "SELECT status, COUNT(*) FROM jobs WHERE workspace = ? GROUP BY status", (workspace,)
        ).fetchall()
        models = connection.execute(
            "SELECT execution_model, COUNT(*) FROM jobs WHERE workspace = ? "
            "GROUP BY execution_model",
            (workspace,),
        ).fetchall()
        # Unreaped reservations occupy capacity, so this list is bounded by it.
        unreaped = connection.execute(
            "SELECT r.job_id FROM admission_reservations r JOIN jobs j ON j.job_id = r.job_id "
            "WHERE r.state = 'unreaped' AND j.workspace = ? ORDER BY r.job_id",
            (workspace,),
        ).fetchall()
    by_status: dict[str, JsonValue] = {str(row[0]): int(row[1]) for row in statuses}
    by_model: dict[str, JsonValue] = {str(row[0]): int(row[1]) for row in models}
    return {
        "statuses": by_status,
        "execution_models": by_model,
        "pending_purge_count": ledger.pending_purge_count(),
        "unreaped": [str(row[0]) for row in unreaped],
        "admission": _plain(admission.snapshot().to_public_dict()),
    }


def _response(
    request: StorageQueryRequest,
    status: QueryStatus,
    *,
    items: list[dict[str, JsonValue]] | None = None,
    summary: dict[str, JsonValue] | None = None,
    next_after: str | None = None,
) -> StorageQueryResponse:
    return StorageQueryResponse(
        schema_version=QUERY_SCHEMA_VERSION,
        operation="query",
        request_id=request.request_id,
        view=request.view,
        status=status,
        items=tuple(items or ()),
        summary=summary,
        next_after=next_after,
    )


def apply_query(
    ledger: StudioJobLedger,
    admission: SharedJobAdmission,
    request: StorageQueryRequest,
    *,
    authorize: Callable[[StorageQueryRequest], bool],
    max_bytes: int,
) -> bytes:
    """Answer one decoded request whose workspace matched the service.

    Returns
    -------
    bytes
        The encoded response, within ``max_bytes``.
    """
    if not authorize(request):
        return encode_query_message(_response(request, "forbidden"))
    if request.view == "status":
        summary = _summary(ledger, admission, request.workspace)
        return encode_query_message(_response(request, "ok", summary=summary))
    page = (
        read_record_page(ledger, request, max_bytes=max_bytes)
        if request.view == "records"
        else _purges(ledger, request)
    )
    if page is None:
        return encode_query_message(_response(request, "invalid_cursor"))
    items, more = page
    return fit_page(request, items, more, max_bytes=max_bytes)


def serve_query(
    channel: socket.socket,
    *,
    ledger: StudioJobLedger,
    admission: SharedJobAdmission,
    gateway: PolicyGateway,
    workspace: str,
    expected_api_uid: int,
    max_bytes: int,
    deadline: float,
    initial_frame: bytes | None = None,
    max_content_bytes: int | None = None,
) -> None:
    """Serve one peer-verified query after the route's policy allowed it.

    Parameters
    ----------
    channel : socket.socket
        Connected Unix stream, closed by this handler on every outcome.
    ledger : StudioJobLedger
        Existing authority; read only.
    admission : SharedJobAdmission
        The service's configured admission, for occupancy and limits.
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
    max_content_bytes : int, optional
        Independent total page ceiling; defaults to the event custody ceiling
        plus one frame. Oversized records are refused without truncation.

    Raises
    ------
    ValueError
        Configuration, request or workspace is invalid; nothing is answered.
    PermissionError
        The peer is not the configured API.
    TimeoutError, EOFError, OSError
        Wire transfer fails.
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
        request = decode_query_request(frame, max_bytes=max_bytes)
        if request.workspace != workspace:
            raise ValueError("storage request does not match configured workspace")
        registry = build_default_studio_route_policy_registry()

        def authorize(query: StorageQueryRequest) -> bool:
            requester = query.requester
            principal = (
                None
                if requester is None
                else Principal(requester.principal_id, frozenset(requester.roles))
            )
            route = QUERY_ROUTES[query.view]
            decision = gateway.authorize(
                registry.policy_for("GET", route),
                principal=principal,
                route=route,
                request_id=query.request_id,
            )
            return decision.allowed

        content_limit = view_content_limit(max_bytes, max_content_bytes)
        response = apply_query(
            ledger, admission, request, authorize=authorize, max_bytes=content_limit
        )
        send_view_content(
            channel,
            response,
            content_schema=QUERY_SCHEMA_VERSION,
            request_id=request.request_id,
            expected_uid=expected_api_uid,
            frame_max_bytes=max_bytes,
            deadline=deadline,
            max_content_bytes=content_limit,
        )


__all__ = ["apply_query", "fit_page", "serve_query"]
