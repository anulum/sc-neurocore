# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — named storage admission request contract

"""Decode exact bounded admission metadata before any seed or ledger access."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester

_Name = Annotated[str, Field(min_length=1)]
_SeedSize = Annotated[int, Field(ge=0)]
_ExecutionTimeout = Annotated[float, Field(gt=0)]
_QueueWait = Annotated[float, Field(ge=0)]


class StorageNamedAdmissionRequest(BaseModel):
    """One named submission from a verified API peer, without worker authority.

    The service separately checks the configured workspace, route policy,
    reviewed task registry, delegated requester and peer process generation.
    ``seed_manifest`` describes frames following this metadata request; the
    service checks actual bytes before deriving durable replay identity.
    No client-selected kind, owner, import path, digest or supervisor is valid.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    schema_version: Literal["studio.storage.admission.v1"]
    operation: Literal["admit_named"]
    request_id: str | None
    mutation_id: _Name
    workspace: _Name
    requester: StorageRequester | None
    task_name: _Name
    authorized_route: _Name
    payload: dict[str, object]
    seed_manifest: dict[str, _SeedSize]
    execution_timeout_seconds: _ExecutionTimeout
    queue_wait_seconds: _QueueWait | None
    admission: dict[str, object] | None
    training_config: dict[str, object] | None
    experiment_sha256: str | None


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate storage admission field")
        fields[name] = value
    return fields


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite storage admission JSON constant")


def decode_named_admission_request(
    payload: bytes, *, max_metadata_bytes: int
) -> StorageNamedAdmissionRequest:
    """Decode one exact metadata frame with no ambiguous JSON object names.

    Parameters
    ----------
    payload : bytes
        Complete nonempty metadata frame from a peer-verified Unix stream.
    max_metadata_bytes : int
        Service-configured positive frame ceiling checked before parsing.

    Returns
    -------
    StorageNamedAdmissionRequest
        Strictly typed metadata, not an authorization or admission decision.

    Raises
    ------
    ValueError
        Bytes, JSON, schema version, operation, field types or size are invalid.

    Notes
    -----
    The caller must still verify the configured API UID and process generation,
    authorize the route, receive exact seed bytes and establish worker custody
    before durable admission. This decoder has no ledger or task import access.
    """
    if (
        type(max_metadata_bytes) is not int
        or max_metadata_bytes <= 0
        or not isinstance(payload, bytes)
        or not 0 < len(payload) <= max_metadata_bytes
    ):
        raise ValueError("storage admission metadata exceeds byte limit")
    try:
        text = payload.decode("utf-8")
        json.loads(
            text,
            object_pairs_hook=_unique_fields,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid storage admission metadata JSON") from exc
    return StorageNamedAdmissionRequest.model_validate_json(text, strict=True)
