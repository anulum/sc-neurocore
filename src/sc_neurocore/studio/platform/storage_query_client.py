# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API client for storage authority views

"""Read the storage authority's views from the trusted API.

Reads have no side effect, so a lost reply is simply read again on a new
connection. Every record page is decoded into complete native snapshots of
the configured workspace; a page whose cursor does not name its own last
item is refused, so a malformed authority cannot make the API loop.
"""

from __future__ import annotations

from collections.abc import Callable
import secrets
import socket
import time

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from sc_neurocore.studio.platform.jobs_models import (
    StudioJobPurgeRecord,
    StudioJobPurgeSnapshot,
    StudioJobRecord,
    StudioJobResourceProfile,
    StudioJobStatusSnapshot,
)
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_query_protocol import (
    QUERY_SCHEMA_VERSION,
    QueryView,
    StorageQueryRequest,
    StorageQueryResponse,
    decode_query_response,
    encode_query_message,
)
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester


def query_request(
    workspace: str,
    view: QueryView,
    *,
    requester: StorageRequester | None,
    limit: int = 1000,
    after: str | None = None,
) -> StorageQueryRequest:
    """Build a query with a fresh random request ID.

    Raises
    ------
    pydantic.ValidationError
        A field does not match the wire grammar.
    """
    return StorageQueryRequest(
        schema_version=QUERY_SCHEMA_VERSION,
        operation="query",
        request_id=secrets.token_hex(16),
        workspace=workspace,
        requester=requester,
        view=view,
        limit=limit,
        after=after,
    )


def exchange_query(
    channel: socket.socket,
    request: StorageQueryRequest,
    *,
    expected_service_uid: int,
    max_bytes: int,
    deadline: float,
) -> StorageQueryResponse:
    """Run one query over a connected, exclusively owned stream.

    Raises
    ------
    PermissionError
        The peer is not the configured storage identity, or policy denied.
    ValueError
        The reply is malformed, answers another request or refused the cursor.
    TimeoutError, EOFError, OSError
        The exchange failed; reading again is safe.
    """
    with channel:
        write_verified_frame(
            channel,
            encode_query_message(request),
            expected_uid=expected_service_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )
        reply = read_verified_frame(
            channel, expected_uid=expected_service_uid, max_bytes=max_bytes, deadline=deadline
        )
    return decode_query_response(reply, request=request, max_bytes=max_bytes)


class QueryReader:
    """Read whole views through a connection factory and bounded exchanges."""

    def __init__(
        self,
        connect: Callable[[], socket.socket],
        *,
        workspace: str,
        storage_uid: int,
        max_bytes: int,
        timeout_seconds: float,
    ) -> None:
        """Keep the trusted endpoint settings; nothing is sent yet."""
        self._connect = connect
        self._workspace = workspace
        self._storage_uid = storage_uid
        self._max_bytes = max_bytes
        self._timeout = timeout_seconds

    def _query(
        self, view: QueryView, requester: StorageRequester | None, *, limit: int, after: str | None
    ) -> StorageQueryResponse:
        request = query_request(
            self._workspace, view, requester=requester, limit=limit, after=after
        )
        return exchange_query(
            self._connect(),
            request,
            expected_service_uid=self._storage_uid,
            max_bytes=self._max_bytes,
            deadline=time.monotonic() + self._timeout,
        )

    def records(self, requester: StorageRequester | None) -> tuple[StudioJobRecord, ...]:
        """Return every record of the workspace in creation order.

        Raises
        ------
        ValueError
            A page is malformed, names another workspace, or its cursor does
            not name its own last record.
        """
        records: list[StudioJobRecord] = []
        after: str | None = None
        while True:
            page = self._query("records", requester, limit=1000, after=after)
            decoded = [decode_job_snapshot(item) for item in page.items]
            if any(record.workspace != self._workspace for record in decoded):
                raise ValueError("storage record page names another workspace")
            records.extend(decoded)
            if page.next_after is None:
                return tuple(records)
            if not decoded or page.next_after != decoded[-1].job_id:
                raise ValueError("storage record page cursor does not advance")
            after = page.next_after

    def status(self, requester: StorageRequester | None) -> StatusSummary:
        """Return the authority's aggregate summary for the workspace.

        Raises
        ------
        pydantic.ValidationError
            The summary does not have the authority's exact shape.
        """
        summary = self._query("status", requester, limit=1, after=None).summary
        return StatusSummary.model_validate(summary, strict=True)

    def purges(
        self, requester: StorageRequester | None, *, limit: int, after: str | None
    ) -> StudioJobPurgeSnapshot:
        """Return one operator purge journal page."""
        page = self._query("purges", requester, limit=limit, after=after)
        purges = tuple(
            StudioJobPurgeRecord(**_PurgeItem.model_validate(item, strict=True).model_dump())
            for item in page.items
        )
        return StudioJobPurgeSnapshot(purges=purges, next_after=page.next_after)


class StatusSummary(BaseModel):
    """The authority's workspace summary, exactly as it serialises it."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    statuses: dict[str, Annotated[int, Field(ge=0)]]
    execution_models: dict[str, Annotated[int, Field(ge=0)]]
    pending_purge_count: Annotated[int, Field(ge=0)]
    unreaped: list[Annotated[str, Field(pattern=r"^sj_[0-9a-f]{16}$")]]
    admission: dict[str, Annotated[int, Field(ge=0)]]

    def snapshot(
        self,
        *,
        allowed_kinds: frozenset[str],
        default_timeout_seconds: float,
        max_artifact_bytes: int,
    ) -> StudioJobStatusSnapshot:
        """Return the embedded status shape for this summary.

        The isolated profile runs process jobs only and keeps no local
        recovery decisions, so those fields describe exactly that.
        """
        statuses, models = self.statuses, self.execution_models
        kinds = tuple(sorted(allowed_kinds))
        return StudioJobStatusSnapshot(
            configured=True,
            allowed_kinds=kinds,
            active_count=sum(
                statuses.get(name, 0) for name in ("pending", "running", "cancelling")
            ),
            completed_count=statuses.get("completed", 0),
            failed_count=statuses.get("failed", 0),
            process_count=models.get("process", 0),
            thread_count=models.get("thread", 0),
            timed_out_count=statuses.get("timed_out", 0),
            interrupted_count=statuses.get("interrupted", 0),
            unknown_count=statuses.get("unknown", 0),
            admission=dict(self.admission),
            unreaped_workers=tuple(self.unreaped),
            pending_purge_count=self.pending_purge_count,
            resource_profiles=tuple(
                StudioJobResourceProfile(
                    kind=kind,
                    default_timeout_seconds=default_timeout_seconds,
                    max_artifact_bytes=max_artifact_bytes,
                    execution_models=("process",),
                )
                for kind in kinds
            ),
        )


class _PurgeItem(BaseModel):
    """One purge journal entry exactly as the authority serialises it."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    job_id: Annotated[str, Field(pattern=r"^sj_[0-9a-f]{16}$")]
    phase: Annotated[str, Field(min_length=1)]
    device: int | None
    inode: int | None


__all__ = ["QueryReader", "StatusSummary", "exchange_query", "query_request"]
