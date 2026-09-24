# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage purge wire contract

"""Strict wire contract for purging one terminal job at the authority.

``studio.storage.purge.v1`` asks the authority to purge one terminal,
unreserved job of the workspace with its sealed directory, for the requester
the peer-verified API delegated, under the policy of the archive purge route,
the one route that purges. The reply carries the purged record, or the
ledger's refusal as text, as the embedded manager raises it. A lost reply is
resolved by reading the record: a purged job is not found.
"""

from __future__ import annotations

import json
from typing import Annotated, Final, Literal
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester

PURGE_SCHEMA_VERSION: Final[Literal["studio.storage.purge.v1"]] = "studio.storage.purge.v1"
PURGE_ROUTE: Final = ("POST", "/api/studio/audit/quarantine/archive/purge")

_RequestId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_JobId = Annotated[str, Field(pattern=r"^sj_[0-9a-f]{16}$")]


class StoragePurgeRequest(BaseModel):
    """Purge one terminal job of the configured workspace."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.purge.v1"]
    operation: Literal["purge"]
    request_id: _RequestId
    workspace: Annotated[str, Field(min_length=1, max_length=256)]
    requester: StorageRequester | None
    job_id: _JobId


class StoragePurgeResponse(BaseModel):
    """The purged record, a refusal text, or a fixed status."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.purge.v1"]
    operation: Literal["purge"]
    request_id: _RequestId
    job_id: _JobId
    status: Literal["ok", "forbidden", "not_found", "refused"]
    record: dict[str, JsonValue] | None
    error: Annotated[str, Field(min_length=1, max_length=512)] | None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """A purge carries the record; a refusal carries the ledger's reason."""
        if (self.status == "ok") != (self.record is not None):
            raise ValueError("only a purge carries its record")
        if (self.status == "refused") != (self.error is not None):
            raise ValueError("only a refused purge carries an error")
        return self


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON names so no field is chosen ambiguously."""
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate storage purge field")
        fields[name] = value
    return fields


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON extensions."""
    raise ValueError("nonfinite storage purge constant")


def _checked_text(payload: bytes, max_bytes: int) -> str:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= max_bytes:
        raise ValueError("storage purge frame exceeds byte limit")
    try:
        text = payload.decode("utf-8")
        json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid storage purge JSON") from exc
    return text


def encode_purge_message(message: StoragePurgeRequest | StoragePurgeResponse) -> bytes:
    """Serialise a validated message as compact, sorted UTF-8 JSON."""
    return json.dumps(
        message.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def decode_purge_request(payload: bytes, *, max_bytes: int) -> StoragePurgeRequest:
    """Decode one exact request frame.

    Raises
    ------
    ValueError
        Size, encoding, duplicate names, unknown fields or any field is invalid.
    """
    return StoragePurgeRequest.model_validate_json(_checked_text(payload, max_bytes), strict=True)


def decode_purge_response(
    payload: bytes, *, request: StoragePurgeRequest, max_bytes: int
) -> StoragePurgeResponse:
    """Decode a response and require that it answers ``request``.

    Raises
    ------
    ValueError
        The frame is malformed or answers another request or job.
    PermissionError
        The authority's policy denied the requester.
    KeyError
        The job is not in the workspace.
    StudioJobRejected
        The ledger refused the purge; the message is the ledger's.
    """
    response = StoragePurgeResponse.model_validate_json(
        _checked_text(payload, max_bytes), strict=True
    )
    if response.request_id != request.request_id or response.job_id != request.job_id:
        raise ValueError("storage purge response does not answer the request")
    if response.status == "forbidden":
        raise PermissionError("storage purge denied")
    if response.status == "not_found":
        raise KeyError(request.job_id)
    if response.status == "refused":
        raise StudioJobRejected(str(response.error))
    return response


__all__ = [
    "PURGE_ROUTE",
    "PURGE_SCHEMA_VERSION",
    "StoragePurgeRequest",
    "StoragePurgeResponse",
    "decode_purge_request",
    "decode_purge_response",
    "encode_purge_message",
]
