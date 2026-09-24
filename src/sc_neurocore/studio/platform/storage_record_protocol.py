# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — exact storage record request schema

"""Strict record-read wire input; delegated claims are not authentication.

Only a separately verified trusted API connection may supply a requester.
The service must still apply its existing policy and server-bound workspace.
Decode only payloads already limited by the storage framing layer.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot

_Name = Annotated[str, Field(min_length=1)]


class StorageRequester(BaseModel):
    """Exact delegated principal claim from the configured trusted API peer."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    principal_id: _Name
    roles: tuple[_Name, ...]


class StorageRecordRequest(BaseModel):
    """Versioned read-only operation, keeping trace and requester separate."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.record.v2"]
    operation: Literal["record"]
    request_id: str | None
    job_id: _Name
    workspace: _Name
    requester: StorageRequester | None


class StorageRecordResponse(BaseModel):
    """Exact read outcome; success requires a complete domain snapshot."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.record.v2"]
    request_id: str | None
    status: Literal["ok", "forbidden", "not_found"]
    record: dict[str, object] | None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate storage record field")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite storage JSON constant")


def _validated_text(payload: bytes) -> str:
    try:
        text = payload.decode("utf-8")
        json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        return text
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid storage record JSON") from exc


def decode_record_request(payload: bytes) -> StorageRecordRequest:
    """Decode a bounded exact request without accepting ambiguous JSON objects.

    Parameters
    ----------
    payload : bytes
        UTF-8 JSON payload already bounded by the peer-verified frame reader.

    Returns
    -------
    StorageRecordRequest
        Strict typed request, not an authorization decision.

    Raises
    ------
    ValueError
        Encoding, nesting, duplicate keys, constants, fields or types are invalid.

    Notes
    -----
    Nullable fields remain mandatory. No defaults silently repair a request.
    Workspace and requester checks belong to the authority before ledger access.
    """
    return StorageRecordRequest.model_validate_json(_validated_text(payload), strict=True)


def decode_record_response(payload: bytes, *, request: StorageRecordRequest) -> StudioJobRecord:
    """Reconstruct one complete correlated record from a bounded response.

    Parameters
    ----------
    payload : bytes
        Complete bounded UTF-8 response from the verified storage peer.
    request : StorageRecordRequest
        Original request, supplying expected trace, job and workspace.

    Returns
    -------
    StudioJobRecord
        Complete native snapshot; no ledger is constructed on the client.

    Raises
    ------
    ValueError
        Wire schema, JSON, correlation, outcome or snapshot is inconsistent.
    PermissionError
        Authority denied the read; no record was supplied.
    KeyError
        Authority found no record in the requested workspace.

    Notes
    -----
    Trace matching is not replay authentication. The connected peer is verified
    separately, and errors never expose a partial or mismatched record.
    """
    response = StorageRecordResponse.model_validate_json(_validated_text(payload), strict=True)
    if response.request_id != request.request_id:
        raise ValueError("storage response trace does not match request")
    if response.status != "ok":
        if response.record is not None:
            raise ValueError("storage error response contains a record")
        if response.status == "forbidden":
            raise PermissionError("storage record access denied")
        raise KeyError(request.job_id)
    if response.record is None:
        raise ValueError("storage success response has no record")
    record = decode_job_snapshot(response.record)
    if record.job_id != request.job_id or record.workspace != request.workspace:
        raise ValueError("storage response record does not match request")
    return record
