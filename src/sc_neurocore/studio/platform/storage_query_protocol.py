# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage query wire contract

"""Strict wire contract for bounded read-only views of the storage authority.

``studio.storage.query.v1`` carries one of three views, each governed by the
policy of the HTTP route it serves: ``records`` (a page of the workspace's job
records in creation order), ``status`` (aggregate counts, admission occupancy
and pending purges) and ``purges`` (the operator purge journal page). The
requester is the API's authenticated principal, delegated only by the
peer-verified API; the authority applies the existing policy before reading.
A page never exceeds the frame: the authority returns fewer items and a
cursor instead.
"""

from __future__ import annotations

import json
from typing import Annotated, Final, Literal
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester

QUERY_SCHEMA_VERSION: Final[Literal["studio.storage.query.v1"]] = "studio.storage.query.v1"

_RequestId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_JobId = Annotated[str, Field(pattern=r"^sj_[0-9a-f]{16}$")]
_Workspace = Annotated[str, Field(min_length=1, max_length=256)]

QueryView = Literal["records", "status", "purges"]
QueryStatus = Literal["ok", "forbidden", "invalid_cursor"]
QUERY_ROUTES: Final[dict[str, str]] = {
    "records": "/api/studio/jobs",
    "status": "/api/studio/jobs/status",
    "purges": "/api/studio/jobs/purges",
}


class StorageQueryRequest(BaseModel):
    """One bounded read; ``status`` takes no cursor."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.query.v1"]
    operation: Literal["query"]
    request_id: _RequestId
    workspace: _Workspace
    requester: StorageRequester | None
    view: QueryView
    limit: Annotated[int, Field(ge=1, le=1000)]
    after: _JobId | None

    @model_validator(mode="after")
    def validate_cursor(self) -> Self:
        """Refuse a cursor on the aggregate view."""
        if self.view == "status" and self.after is not None:
            raise ValueError("the status view takes no cursor")
        return self


class StorageQueryResponse(BaseModel):
    """The authority's answer; items and summary belong to the requested view."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.query.v1"]
    operation: Literal["query"]
    request_id: _RequestId
    view: QueryView
    status: QueryStatus
    items: tuple[dict[str, JsonValue], ...]
    summary: dict[str, JsonValue] | None
    next_after: _JobId | None

    @model_validator(mode="after")
    def validate_view(self) -> Self:
        """Only an answered page carries items or a cursor; only status a summary."""
        answered = self.status == "ok"
        if not answered and (self.items or self.summary is not None or self.next_after):
            raise ValueError("a refused query carries no data")
        if answered and (self.view == "status") != (self.summary is not None):
            raise ValueError("only the status view carries a summary")
        if self.view == "status" and (self.items or self.next_after is not None):
            raise ValueError("the status view carries no items")
        return self


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON names so no field is chosen ambiguously."""
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate storage query field")
        fields[name] = value
    return fields


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON extensions."""
    raise ValueError("nonfinite storage query constant")


def _checked_text(payload: bytes, max_bytes: int) -> str:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= max_bytes:
        raise ValueError("storage query frame exceeds byte limit")
    try:
        text = payload.decode("utf-8")
        json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid storage query JSON") from exc
    return text


def encode_query_message(message: StorageQueryRequest | StorageQueryResponse) -> bytes:
    """Serialise a validated message as compact, sorted UTF-8 JSON."""
    return json.dumps(
        message.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def decode_query_request(payload: bytes, *, max_bytes: int) -> StorageQueryRequest:
    """Decode one exact request frame.

    Raises
    ------
    ValueError
        Size, encoding, duplicate names, unknown fields or any field is invalid.
    """
    return StorageQueryRequest.model_validate_json(_checked_text(payload, max_bytes), strict=True)


def decode_query_response(
    payload: bytes, *, request: StorageQueryRequest, max_bytes: int
) -> StorageQueryResponse:
    """Decode a response and require that it answers ``request``.

    Raises
    ------
    ValueError
        The frame is malformed or answers another request or view, or the
        cursor was refused.
    PermissionError
        The authority's policy denied the requester.
    """
    response = StorageQueryResponse.model_validate_json(
        _checked_text(payload, max_bytes), strict=True
    )
    if response.request_id != request.request_id or response.view != request.view:
        raise ValueError("storage query response does not answer the request")
    if response.status == "forbidden":
        raise PermissionError("storage query access denied")
    if response.status == "invalid_cursor":
        raise ValueError("storage query cursor is not a job of this view")
    return response


__all__ = [
    "QUERY_ROUTES",
    "QUERY_SCHEMA_VERSION",
    "QueryStatus",
    "QueryView",
    "StorageQueryRequest",
    "StorageQueryResponse",
    "decode_query_request",
    "decode_query_response",
    "encode_query_message",
]
