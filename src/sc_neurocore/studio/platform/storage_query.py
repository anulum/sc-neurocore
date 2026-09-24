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
from sc_neurocore.studio.platform.jobs_ledger_schema import record_from_row
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.policy_routes import build_default_studio_route_policy_registry
from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_supervisor_identity,
    write_verified_frame,
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


def _records(ledger: StudioJobLedger, request: StorageQueryRequest) -> _Page | None:
    """Read one creation-ordered page; ``None`` when the cursor is unknown."""
    with ledger.transaction() as connection:
        cursor: tuple[str, int] = ("", 0)
        if request.after is not None:
            row = connection.execute(
                "SELECT created_at_utc, rowid FROM jobs WHERE job_id = ? AND workspace = ?",
                (request.after, request.workspace),
            ).fetchone()
            if row is None:
                return None
            cursor = (str(row[0]), int(row[1]))
        rows = connection.execute(
            "SELECT * FROM jobs WHERE workspace = ? AND (created_at_utc, rowid) > (?, ?) "
            "ORDER BY created_at_utc, rowid LIMIT ?",
            (request.workspace, *cursor, request.limit + 1),
        ).fetchall()
    items = [record_from_row(row).to_public_dict() for row in rows[: request.limit]]
    return [_plain(item) for item in items], len(rows) > request.limit


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


def fit_page(
    request: StorageQueryRequest,
    items: list[dict[str, JsonValue]],
    more: bool,
    *,
    max_bytes: int,
) -> bytes:
    """Encode the longest prefix of ``items`` whose response fits the frame.

    Each item is measured by its own compact encoding plus one separator, and
    the envelope is measured with a cursor of full length, so the estimate
    never undercounts the encoded page. A shortened page carries a cursor to
    its last item, so the API continues where it stopped.

    Raises
    ------
    ValueError
        A single item does not fit the frame: a configuration fault.
    """
    placeholder = "sj_" + "0" * 16
    budget = max_bytes - len(
        encode_query_message(_response(request, "ok", items=[], next_after=placeholder))
    )
    count = 0
    for item in items:
        size = len(json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)) + 1
        if size > budget:
            break
        budget -= size
        count += 1
    if items and count == 0:
        raise ValueError("one storage query item exceeds the frame limit")
    page = items[:count]
    cursor = str(page[-1]["job_id"]) if page and (more or count < len(items)) else None
    return encode_query_message(_response(request, "ok", items=page, next_after=cursor))


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
    page = _records(ledger, request) if request.view == "records" else _purges(ledger, request)
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

        response = apply_query(ledger, admission, request, authorize=authorize, max_bytes=max_bytes)
        write_verified_frame(
            channel, response, expected_uid=expected_api_uid, max_bytes=max_bytes, deadline=deadline
        )


__all__ = ["apply_query", "fit_page", "serve_query"]
