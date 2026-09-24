# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority purge

"""Purge one terminal job and its sealed directory at the storage authority.

The authority runs the embedded purge with its durable phases, staging and
recovery over its own root: the sealed ``<root>/<job_id>`` directory is
staged beside itself, the record deleted and the stage removed only with
committed evidence. The authority supervises no workers, so it has no local
handles to forget. A refusal (active, reserved or pending purge) is returned
with the ledger's own message.
"""

from __future__ import annotations

import json
from pathlib import Path
import socket
import threading

from pydantic import JsonValue

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.jobs_paths import _resolve_job_directory
from sc_neurocore.studio.platform.jobs_purge import purge_terminal_job
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.policy_routes import build_default_studio_route_policy_registry
from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_supervisor_identity,
    write_verified_frame,
)
from sc_neurocore.studio.platform.storage_purge_protocol import (
    PURGE_ROUTE,
    PURGE_SCHEMA_VERSION,
    StoragePurgeRequest,
    StoragePurgeResponse,
    decode_purge_request,
    encode_purge_message,
)


class AuthorityCustody:
    """The authority as a purge owner: its ledger and root, no local handles."""

    def __init__(self, ledger: StudioJobLedger) -> None:
        """Bind purging to the authority's ledger and the root that holds it."""
        self._ledger = ledger
        self._root: Path = ledger.path.parent
        self._lock = threading.Lock()
        self._done_events: dict[str, threading.Event] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._unreaped_workers: set[str] = set()

    def _job_work_dir(self, job_id: str) -> Path:
        """Return the sealed directory of one job under the authority root."""
        return _resolve_job_directory(
            root=self._root, job_id=job_id, error_message="Studio job path escapes the job root."
        )


def apply_purge(
    custody: AuthorityCustody, request: StoragePurgeRequest, *, allowed: bool
) -> StoragePurgeResponse:
    """Apply one decoded request whose workspace matched the service."""
    status, error = "forbidden", None
    record: dict[str, JsonValue] | None = None
    if allowed:
        try:
            custody._ledger.record(request.job_id, workspace=request.workspace)
            purged = purge_terminal_job(custody, request.job_id)
        except KeyError:
            status = "not_found"
        except StudioJobRejected as refused:
            status, error = "refused", str(refused)[:512] or "refused"
        else:
            status = "ok"
            record = json.loads(json.dumps(purged.to_public_dict(), allow_nan=False))
    return StoragePurgeResponse.model_validate(
        {
            "schema_version": PURGE_SCHEMA_VERSION,
            "operation": "purge",
            "request_id": request.request_id,
            "job_id": request.job_id,
            "status": status,
            "record": record,
            "error": error,
        },
        strict=True,
    )


def serve_purge(
    channel: socket.socket,
    *,
    custody: AuthorityCustody,
    gateway: PolicyGateway,
    workspace: str,
    expected_api_uid: int,
    max_bytes: int,
    deadline: float,
    initial_frame: bytes | None = None,
) -> None:
    """Serve one peer-verified purge after the archive purge route's policy.

    Parameters
    ----------
    channel : socket.socket
        Connected Unix stream, closed by this handler on every outcome.
    custody : AuthorityCustody
        The authority's ledger and root.
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
        Configuration, request or workspace is invalid; nothing is purged.
    PermissionError
        The peer is not the configured API.
    TimeoutError, EOFError, OSError
        Wire transfer fails; read the record to learn whether it was purged.
    AuditSinkError
        The policy audit cannot persist its decision; nothing is purged.
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
        request = decode_purge_request(frame, max_bytes=max_bytes)
        if request.workspace != workspace:
            raise ValueError("storage request does not match configured workspace")
        requester = request.requester
        principal = (
            None
            if requester is None
            else Principal(requester.principal_id, frozenset(requester.roles))
        )
        method, route = PURGE_ROUTE
        decision = gateway.authorize(
            build_default_studio_route_policy_registry().policy_for(method, route),
            principal=principal,
            route=route,
            request_id=request.request_id,
        )
        response = apply_purge(custody, request, allowed=decision.allowed)
        write_verified_frame(
            channel,
            encode_purge_message(response),
            expected_uid=expected_api_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )


__all__ = ["AuthorityCustody", "apply_purge", "serve_purge"]
