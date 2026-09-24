# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sealed artefact read wire contract

"""Strict wire contract for reading one sealed artefact from the authority.

``studio.storage.artifact.v1`` names the HTTP route the read serves, from the
closed set of routes that read completed artefacts, so the authority applies
that route's policy to the delegated requester. An answered read carries the
declared artefact, and its bytes follow in one frame when it is not empty;
the isolated profile seals only artefacts that fit one frame.
"""

from __future__ import annotations

import json
from typing import Annotated, Final, Literal
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sc_neurocore.studio.platform.jobs_models import StudioJobArtifactUnavailable
from sc_neurocore.studio.platform.storage_finish_protocol import FinishArtifact
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester

ARTIFACT_SCHEMA_VERSION: Final[Literal["studio.storage.artifact.v1"]] = "studio.storage.artifact.v1"

ArtifactRoute = Literal[
    "/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}",
    "/api/studio/training/weight-restore",
    "/api/studio/training/weight-restore/attach",
    "/api/studio/training/weight-restore/attach/live",
]
ARTIFACT_ROUTE_METHODS: Final[dict[str, str]] = {
    "/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}": "GET",
    "/api/studio/training/weight-restore": "POST",
    "/api/studio/training/weight-restore/attach": "POST",
    "/api/studio/training/weight-restore/attach/live": "POST",
}

_RequestId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_JobId = Annotated[str, Field(pattern=r"^sj_[0-9a-f]{16}$")]


class StorageArtifactRequest(BaseModel):
    """Read one declared artefact of a job in the configured workspace."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.artifact.v1"]
    operation: Literal["artifact"]
    request_id: _RequestId
    workspace: Annotated[str, Field(min_length=1, max_length=256)]
    requester: StorageRequester | None
    route: ArtifactRoute
    job_id: _JobId
    relative_path: Annotated[str, Field(min_length=1, max_length=512)]


class StorageArtifactResponse(BaseModel):
    """The declared artefact, or a fixed refusal."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.artifact.v1"]
    operation: Literal["artifact"]
    request_id: _RequestId
    status: Literal["ok", "forbidden", "not_found", "unavailable"]
    artifact: FinishArtifact | None

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        """Exactly an answered read carries the artefact."""
        if (self.status == "ok") != (self.artifact is not None):
            raise ValueError("only an answered read carries an artefact")
        return self


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON names so no field is chosen ambiguously."""
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate storage artefact field")
        fields[name] = value
    return fields


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON extensions."""
    raise ValueError("nonfinite storage artefact constant")


def _checked_text(payload: bytes, max_bytes: int) -> str:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= max_bytes:
        raise ValueError("storage artefact frame exceeds byte limit")
    try:
        text = payload.decode("utf-8")
        json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid storage artefact JSON") from exc
    return text


def encode_artifact_message(message: StorageArtifactRequest | StorageArtifactResponse) -> bytes:
    """Serialise a validated message as compact, sorted UTF-8 JSON."""
    return json.dumps(
        message.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def decode_artifact_request(payload: bytes, *, max_bytes: int) -> StorageArtifactRequest:
    """Decode one exact request frame.

    Raises
    ------
    ValueError
        Size, encoding, duplicate names, unknown fields, route or field is invalid.
    """
    return StorageArtifactRequest.model_validate_json(
        _checked_text(payload, max_bytes), strict=True
    )


def decode_artifact_response(
    payload: bytes, *, request: StorageArtifactRequest, max_bytes: int
) -> StorageArtifactResponse:
    """Decode a response that must answer ``request`` and name its artefact.

    Raises
    ------
    ValueError
        The frame is malformed, answers another request, or names another path.
    PermissionError
        The route's policy denied the requester.
    KeyError
        The job or the declared artefact is not in the workspace.
    StudioJobArtifactUnavailable
        The sealed bytes are missing or failed their integrity check.
    """
    response = StorageArtifactResponse.model_validate_json(
        _checked_text(payload, max_bytes), strict=True
    )
    if response.request_id != request.request_id:
        raise ValueError("storage artefact response does not answer the request")
    if response.status == "forbidden":
        raise PermissionError("storage artefact read denied")
    if response.status == "not_found":
        raise KeyError(request.relative_path)
    if response.status == "unavailable":
        raise StudioJobArtifactUnavailable("Studio job artifact is unavailable.")
    if response.artifact is None or response.artifact.relative_path != request.relative_path:
        raise ValueError("storage artefact response names another artefact")
    return response


__all__ = [
    "ARTIFACT_ROUTE_METHODS",
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactRoute",
    "StorageArtifactRequest",
    "StorageArtifactResponse",
    "decode_artifact_request",
    "decode_artifact_response",
    "encode_artifact_message",
]
