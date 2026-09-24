# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage cancel wire contract

"""Strict wire contract for recording a cancellation request at the authority.

``studio.storage.cancel.v1`` asks the authority to record that one job of the
workspace should stop, for the requester the peer-verified API delegated,
under the policy of ``POST /api/training/stop``, the one route that cancels.
The reply carries the job's record after the request, as the embedded
manager returns it. Cancelling is idempotent, so a lost reply is resent.
"""

from __future__ import annotations

import json
from typing import Annotated, Final, Literal
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester

CANCEL_SCHEMA_VERSION: Final[Literal["studio.storage.cancel.v1"]] = "studio.storage.cancel.v1"
CANCEL_ROUTE: Final = ("POST", "/api/training/stop")

_RequestId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_JobId = Annotated[str, Field(pattern=r"^sj_[0-9a-f]{16}$")]


class StorageCancelRequest(BaseModel):
    """Record that one job of the configured workspace should stop."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.cancel.v1"]
    operation: Literal["cancel"]
    request_id: _RequestId
    workspace: Annotated[str, Field(min_length=1, max_length=256)]
    requester: StorageRequester | None
    job_id: _JobId


class StorageCancelResponse(BaseModel):
    """The job's record after the request, or a fixed refusal."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.cancel.v1"]
    operation: Literal["cancel"]
    request_id: _RequestId
    job_id: _JobId
    status: Literal["ok", "forbidden", "not_found"]
    record: dict[str, JsonValue] | None

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        """Exactly an answered request carries the record."""
        if (self.status == "ok") != (self.record is not None):
            raise ValueError("only an answered cancellation carries a record")
        return self


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON names so no field is chosen ambiguously."""
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate storage cancel field")
        fields[name] = value
    return fields


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON extensions."""
    raise ValueError("nonfinite storage cancel constant")


def _checked_text(payload: bytes, max_bytes: int) -> str:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= max_bytes:
        raise ValueError("storage cancel frame exceeds byte limit")
    try:
        text = payload.decode("utf-8")
        json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid storage cancel JSON") from exc
    return text


def encode_cancel_message(message: StorageCancelRequest | StorageCancelResponse) -> bytes:
    """Serialise a validated message as compact, sorted UTF-8 JSON."""
    return json.dumps(
        message.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def decode_cancel_request(payload: bytes, *, max_bytes: int) -> StorageCancelRequest:
    """Decode one exact request frame.

    Raises
    ------
    ValueError
        Size, encoding, duplicate names, unknown fields or any field is invalid.
    """
    return StorageCancelRequest.model_validate_json(_checked_text(payload, max_bytes), strict=True)


def decode_cancel_response(
    payload: bytes, *, request: StorageCancelRequest, max_bytes: int
) -> StorageCancelResponse:
    """Decode a response and require that it answers ``request``.

    Raises
    ------
    ValueError
        The frame is malformed or answers another request or job.
    PermissionError
        The authority's policy denied the requester.
    KeyError
        The job is not in the workspace.
    """
    response = StorageCancelResponse.model_validate_json(
        _checked_text(payload, max_bytes), strict=True
    )
    if response.request_id != request.request_id or response.job_id != request.job_id:
        raise ValueError("storage cancel response does not answer the request")
    if response.status == "forbidden":
        raise PermissionError("storage cancel denied")
    if response.status == "not_found":
        raise KeyError(request.job_id)
    return response


__all__ = [
    "CANCEL_ROUTE",
    "CANCEL_SCHEMA_VERSION",
    "StorageCancelRequest",
    "StorageCancelResponse",
    "decode_cancel_request",
    "decode_cancel_response",
    "encode_cancel_message",
]
