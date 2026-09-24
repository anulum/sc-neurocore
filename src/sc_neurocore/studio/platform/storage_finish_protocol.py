# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job finish wire contract

"""Bounded finish messages from the trusted API to the storage authority.

After the launcher reports a generation stopped, the API reports the job's
terminal outcome and declares the artefacts the worker left in its spool. The
authority first answers ``ready`` only for the delegated owner of a live job,
or finally with ``already_sealed`` or ``refused``; after ``ready`` the declared
bytes follow as frames in manifest order, and the authority seals them only
when every size and SHA-256 matches. The supervisor identity
is never a wire field, and no path outside the job is expressible.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Annotated, Final, Literal
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from sc_neurocore.studio.platform.jobs_paths import _relative_path_candidate

FINISH_SCHEMA_VERSION: Final[Literal["studio.storage.finish.v1"]] = "studio.storage.finish.v1"

_JobId = Annotated[str, Field(pattern=r"^sj_[0-9a-f]{16}$")]
_RequestId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_Workspace = Annotated[str, Field(min_length=1, max_length=256)]
_Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

FinishOutcome = Literal["completed", "failed", "cancelled", "timed_out"]
FinishReply = Literal["ready", "sealed", "already_sealed", "refused"]
FinishReason = Literal["not_found", "not_owner", "not_live", "worker_live", "conflict", "bytes"]


class FinishArtifact(BaseModel):
    """One declared artefact: a canonical job-relative path, its size and digest."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    relative_path: Annotated[str, Field(min_length=1, max_length=512)]
    size_bytes: Annotated[int, Field(ge=0)]
    sha256: _Digest

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Accept only a printable, canonical path that stays inside the job."""
        if not self.relative_path.isprintable():
            raise ValueError("artefact path must be printable")
        candidate = _relative_path_candidate(
            self.relative_path, error_message="artefact path escapes the job"
        )
        if candidate.as_posix() != self.relative_path:
            raise ValueError("artefact path must be canonical")
        return self


class StorageFinishRequest(BaseModel):
    """Report a terminal outcome and declare the artefacts to seal.

    As in the embedded supervisor, only ``completed`` carries a ``result`` and
    never an ``error``; ``failed`` and ``timed_out`` carry an error and
    ``cancelled`` may. ``worker_reaped`` is false when the launcher could not
    confirm that every process of the generation ended: the job becomes
    terminal but keeps its capacity as unreaped. A completed job was reaped.
    Artefact paths are unique; their bytes follow as frames in the listed order.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.finish.v1"]
    operation: Literal["finish"]
    request_id: _RequestId
    workspace: _Workspace
    job_id: _JobId
    outcome: FinishOutcome
    result: dict[str, JsonValue] | None
    error: Annotated[str, Field(min_length=1, max_length=1024)] | None
    artifacts: tuple[FinishArtifact, ...]
    worker_reaped: bool

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Match result, error and reaping to the outcome; refuse duplicate paths."""
        completed = self.outcome == "completed"
        if completed and (self.error is not None or not self.worker_reaped):
            raise ValueError("a completed outcome carries no error and was reaped")
        if not completed and self.result is not None:
            raise ValueError("only a completed outcome carries a result")
        if self.outcome in ("failed", "timed_out") and self.error is None:
            raise ValueError("a failed or timed-out outcome carries an error")
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(set(paths)) != len(paths):
            raise ValueError("artefact paths must be unique")
        return self


class StorageFinishResponse(BaseModel):
    """The authority's answer to one finish request.

    ``ready`` is interim and asks for the artefact frames; every other reply
    is final.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.finish.v1"]
    operation: Literal["finish"]
    request_id: _RequestId
    job_id: _JobId
    reply: FinishReply
    reason: FinishReason | None

    @model_validator(mode="after")
    def validate_reply(self) -> Self:
        """Only a refusal carries a reason."""
        if (self.reply == "refused") != (self.reason is not None):
            raise ValueError("only a refused finish carries a reason")
        return self


def validate_artifact_budget(
    artifacts: Sequence[FinishArtifact],
    *,
    frame_max_bytes: int,
    max_artifact_bytes: int,
    max_artifact_entries: int,
) -> None:
    """Hold declared artefacts to the trusted budgets before any byte moves.

    Parameters
    ----------
    artifacts : Sequence[FinishArtifact]
        Declared artefacts, from a request or a worker's own manifest.
    frame_max_bytes : int
        Largest single artefact the framed transfer can carry.
    max_artifact_bytes, max_artifact_entries : int
        Aggregate byte and entry budgets from trusted configuration.

    Raises
    ------
    ValueError
        The declaration exceeds a budget.
    """
    if len(artifacts) > max_artifact_entries:
        raise ValueError("artefact manifest exceeds entry limit")
    if any(artifact.size_bytes > frame_max_bytes for artifact in artifacts):
        raise ValueError("artefact exceeds frame limit")
    if sum(artifact.size_bytes for artifact in artifacts) > max_artifact_bytes:
        raise ValueError("artefacts exceed aggregate limit")


def validate_finish_manifest(
    request: StorageFinishRequest,
    *,
    frame_max_bytes: int,
    max_artifact_bytes: int,
    max_artifact_entries: int,
) -> None:
    """Hold a request's manifest to the trusted artefact budgets.

    Raises
    ------
    ValueError
        The manifest exceeds a budget; see :func:`validate_artifact_budget`.
    """
    validate_artifact_budget(
        request.artifacts,
        frame_max_bytes=frame_max_bytes,
        max_artifact_bytes=max_artifact_bytes,
        max_artifact_entries=max_artifact_entries,
    )


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON names so no field is chosen ambiguously."""
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate storage finish field")
        fields[name] = value
    return fields


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON extensions."""
    raise ValueError("nonfinite storage finish constant")


def _checked_text(payload: bytes, max_bytes: int) -> str:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= max_bytes:
        raise ValueError("storage finish frame exceeds byte limit")
    try:
        text = payload.decode("utf-8")
        json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid storage finish JSON") from exc
    return text


def encode_finish_message(message: StorageFinishRequest | StorageFinishResponse) -> bytes:
    """Serialise a validated request or response as sorted compact JSON.

    Parameters
    ----------
    message : StorageFinishRequest or StorageFinishResponse
        Already validated message.

    Returns
    -------
    bytes
        UTF-8 JSON.
    """
    return json.dumps(
        message.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def decode_finish_request(payload: bytes, *, max_bytes: int) -> StorageFinishRequest:
    """Decode one exact request frame from the verified API peer.

    Parameters
    ----------
    payload : bytes
        Complete frame payload.
    max_bytes : int
        Configured frame ceiling.

    Returns
    -------
    StorageFinishRequest
        Strictly typed request; not an ownership decision.

    Raises
    ------
    ValueError
        Size, encoding, duplicate names, unknown fields or any field shape is
        invalid.
    """
    return StorageFinishRequest.model_validate_json(_checked_text(payload, max_bytes), strict=True)


def decode_finish_response(
    payload: bytes, *, request: StorageFinishRequest, max_bytes: int
) -> StorageFinishResponse:
    """Decode a response and require exact correlation with the sent request.

    Parameters
    ----------
    payload : bytes
        Complete frame payload from the verified storage peer.
    request : StorageFinishRequest
        The request this response must answer.
    max_bytes : int
        Configured frame ceiling.

    Returns
    -------
    StorageFinishResponse
        Correlated answer.

    Raises
    ------
    ValueError
        The frame is malformed, inconsistent or answers another request.
    """
    text = _checked_text(payload, max_bytes)
    response = StorageFinishResponse.model_validate_json(text, strict=True)
    if response.request_id != request.request_id or response.job_id != request.job_id:
        raise ValueError("storage finish response does not answer the request")
    return response
