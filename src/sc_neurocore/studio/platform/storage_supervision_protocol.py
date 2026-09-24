# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job supervision wire contract

"""Bounded start/heartbeat messages from the trusted API to the storage authority.

The API supervises a launcher-started worker but cannot write the ledger. It
asks the storage authority to mark an admitted job running and bind the
verified worker generation (``start``) or to renew its lease (``heartbeat``).
The supervisor identity is never a wire field: the authority derives it from
the connection's pidfd-verified peer. The worker identity carried by ``start``
is the API's observation from the grant endpoint and is checked again by the
authority before registration.
"""

from __future__ import annotations

import json
from typing import Annotated, Final, Literal
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

SUPERVISION_SCHEMA_VERSION: Final[Literal["studio.storage.supervision.v1"]] = (
    "studio.storage.supervision.v1"
)

_JobId = Annotated[str, Field(pattern=r"^sj_[0-9a-f]{16}$")]
_RequestId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_Workspace = Annotated[str, Field(min_length=1, max_length=256)]
_Worker = Annotated[str, Field(pattern=r"^[^:\s]{1,253}:[1-9][0-9]{0,9}:[1-9][0-9]{0,19}$")]

SupervisionOperation = Literal["start", "heartbeat"]
SupervisionOutcome = Literal["started", "cancelling", "renewed", "refused"]
SupervisionReason = Literal[
    "not_found", "not_owner", "not_live", "worker_unverified", "worker_conflict"
]


class _SupervisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.supervision.v1"]
    request_id: _RequestId
    workspace: _Workspace
    job_id: _JobId


class SupervisionStartRequest(_SupervisionRequest):
    """Mark an admitted job running and bind the observed worker generation.

    ``worker`` is the ``host:pid:token`` identity the API verified at its grant
    endpoint; the authority checks it again before registration.
    """

    operation: Literal["start"]
    worker: _Worker


class SupervisionHeartbeatRequest(_SupervisionRequest):
    """Renew the delegated owner's lease on a live job."""

    operation: Literal["heartbeat"]


StorageSupervisionRequest = Annotated[
    SupervisionStartRequest | SupervisionHeartbeatRequest, Field(discriminator="operation")
]
_REQUEST_ADAPTER: TypeAdapter[StorageSupervisionRequest] = TypeAdapter(StorageSupervisionRequest)


class StorageSupervisionResponse(BaseModel):
    """The authority's outcome for one supervision request.

    ``started`` answers ``start``; ``renewed`` answers ``heartbeat``;
    ``cancelling`` answers either for a job whose cancellation is recorded, so
    the owning API stops its worker; ``refused`` carries a fixed reason and changed nothing
    except, for a start of a job already cancelling, its recorded start time.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.supervision.v1"]
    operation: SupervisionOperation
    request_id: _RequestId
    job_id: _JobId
    outcome: SupervisionOutcome
    reason: SupervisionReason | None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Match outcome, operation and reason."""
        if (self.outcome == "refused") != (self.reason is not None):
            raise ValueError("only a refused supervision outcome carries a reason")
        if self.outcome == "started" and self.operation != "start":
            raise ValueError("start outcomes answer only a start request")
        if self.outcome == "renewed" and self.operation != "heartbeat":
            raise ValueError("renewal answers only a heartbeat request")
        if self.reason in ("worker_unverified", "worker_conflict") and self.operation != "start":
            raise ValueError("worker refusals answer only a start request")
        return self


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON names so no field is chosen ambiguously."""
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate storage supervision field")
        fields[name] = value
    return fields


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON extensions."""
    raise ValueError("nonfinite storage supervision constant")


def _checked_text(payload: bytes, max_bytes: int) -> str:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= max_bytes:
        raise ValueError("storage supervision frame exceeds byte limit")
    try:
        text = payload.decode("utf-8")
        json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid storage supervision JSON") from exc
    return text


def encode_supervision_message(
    message: SupervisionStartRequest | SupervisionHeartbeatRequest | StorageSupervisionResponse,
) -> bytes:
    """Serialise a validated request or response as sorted compact JSON.

    Parameters
    ----------
    message : SupervisionStartRequest, SupervisionHeartbeatRequest or StorageSupervisionResponse
        Already validated message.

    Returns
    -------
    bytes
        UTF-8 JSON; every field is bounded by the schema.
    """
    return json.dumps(
        message.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def decode_supervision_request(payload: bytes, *, max_bytes: int) -> StorageSupervisionRequest:
    """Decode one exact request frame from the verified API peer.

    Parameters
    ----------
    payload : bytes
        Complete frame payload.
    max_bytes : int
        Configured frame ceiling.

    Returns
    -------
    SupervisionStartRequest or SupervisionHeartbeatRequest
        Strictly typed request selected by ``operation``; not an ownership
        decision.

    Raises
    ------
    ValueError
        Size, encoding, duplicate names, unknown fields or any field shape is
        invalid.
    """
    text = _checked_text(payload, max_bytes)
    return _REQUEST_ADAPTER.validate_json(text, strict=True)


def decode_supervision_response(
    payload: bytes,
    *,
    request: SupervisionStartRequest | SupervisionHeartbeatRequest,
    max_bytes: int,
) -> StorageSupervisionResponse:
    """Decode a response and require exact correlation with the sent request.

    Parameters
    ----------
    payload : bytes
        Complete frame payload from the verified storage peer.
    request : SupervisionStartRequest or SupervisionHeartbeatRequest
        The request this response must answer.
    max_bytes : int
        Configured frame ceiling.

    Returns
    -------
    StorageSupervisionResponse
        Correlated outcome.

    Raises
    ------
    ValueError
        The frame is malformed, inconsistent or answers another request.
    """
    text = _checked_text(payload, max_bytes)
    response = StorageSupervisionResponse.model_validate_json(text, strict=True)
    if (
        response.request_id != request.request_id
        or response.operation != request.operation
        or response.job_id != request.job_id
    ):
        raise ValueError("storage supervision response does not answer the request")
    return response
