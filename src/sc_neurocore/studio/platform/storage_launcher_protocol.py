# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — worker launcher wire contract

"""Bounded launch/stop/status messages between the trusted API and a launcher.

The launcher is a separate privileged mechanism that starts one fixed worker
bootstrap for an admitted job generation. A request names only a job ID, a
fresh launch generation, an operation and an independent request ID. It never
carries a command, module path, environment, unit name, property or file path;
the launcher derives everything else from its own configuration. Responses
report bounded process state and identity, never output or file contents.
A transport timeout is not evidence that a worker did or did not start: the
caller resolves it with ``status`` for the same job generation.
"""

from __future__ import annotations

import json
from typing import Annotated, Final, Literal
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

LAUNCHER_PROTOCOL_VERSION: Final[Literal["studio.launcher.v1"]] = "studio.launcher.v1"
LAUNCHER_MESSAGE_MAX_BYTES = 1024

_JobId = Annotated[str, Field(pattern=r"^sj_[0-9a-f]{16}$")]
_Hex128 = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_Pid = Annotated[int, Field(gt=0, le=0x3FFFFFFF)]
_StartToken = Annotated[str, Field(pattern=r"^[1-9][0-9]{0,19}$")]
# A negative status is the number of the signal that ended the leader.
_ExitStatus = Annotated[int, Field(ge=-64, le=255)]

LauncherOperation = Literal["launch", "stop", "status"]
LauncherState = Literal["running", "stopped", "absent", "refused"]
LauncherReason = Literal["capacity", "conflict", "survivors", "spool", "unavailable"]


class LauncherRequest(BaseModel):
    """One operation for an exact admitted job generation.

    ``generation`` is chosen by the API once per launch attempt and reused for
    retries of that attempt, so a retry can never create a second generation.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    version: Literal["studio.launcher.v1"]
    request_id: _Hex128
    operation: LauncherOperation
    job_id: _JobId
    generation: _Hex128


class LauncherResponse(BaseModel):
    """The launcher's observed state for the requested job generation.

    ``running`` carries the launched worker PID and its process start token.
    ``stopped`` means the launcher confirmed that no process of the generation
    it tracks remains, and carries the leader's exit status as
    :attr:`subprocess.Popen.returncode` reports it. ``absent`` means this launcher has no record of the
    generation. ``refused`` carries a fixed reason; with ``survivors`` a stop
    could not yet confirm termination and the identity is retained.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    version: Literal["studio.launcher.v1"]
    request_id: _Hex128
    operation: LauncherOperation
    job_id: _JobId
    generation: _Hex128
    state: LauncherState
    pid: _Pid | None
    start_token: _StartToken | None
    reason: LauncherReason | None
    exit_status: _ExitStatus | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        """Require the identity and reason fields that belong to each state."""
        identity = (self.pid is None, self.start_token is None)
        if identity not in ((True, True), (False, False)):
            raise ValueError("launcher identity must carry both PID and start token")
        has_identity = self.pid is not None
        if self.state == "running" and (not has_identity or self.reason is not None):
            raise ValueError("running launcher state requires identity only")
        if self.state == "absent" and (
            has_identity or self.reason is not None or self.operation == "launch"
        ):
            raise ValueError("absent launcher state answers only stop or status")
        if self.state == "stopped" and self.reason is not None:
            raise ValueError("stopped launcher state carries no reason")
        if (self.state == "stopped") != (self.exit_status is not None):
            raise ValueError("only a stopped launcher state carries an exit status")
        if self.state == "refused":
            if self.reason is None:
                raise ValueError("refused launcher state requires a reason")
            if (self.reason == "survivors") != has_identity:
                raise ValueError("only survivor refusal retains worker identity")
            if self.reason == "survivors" and self.operation != "stop":
                raise ValueError("survivor refusal belongs to a stop request")
        elif self.state == "running" and self.operation == "stop":
            raise ValueError("stop cannot report a running worker without refusal")
        return self


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON names so no field is chosen ambiguously."""
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate launcher message field")
        fields[name] = value
    return fields


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON extensions."""
    raise ValueError("nonfinite launcher message constant")


def _checked_text(payload: bytes) -> str:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= LAUNCHER_MESSAGE_MAX_BYTES:
        raise ValueError("launcher message exceeds byte limit")
    try:
        text = payload.decode("utf-8")
        json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid launcher message JSON") from exc
    return text


def _encode(message: BaseModel) -> bytes:
    # Every field is pattern- or range-bounded, so the longest valid message
    # stays far below LAUNCHER_MESSAGE_MAX_BYTES; a test pins that bound.
    return json.dumps(
        message.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def encode_launcher_request(request: LauncherRequest) -> bytes:
    """Serialise a validated request as canonical bounded JSON.

    Parameters
    ----------
    request : LauncherRequest
        Already validated request.

    Returns
    -------
    bytes
        Sorted, compact UTF-8 JSON within :data:`LAUNCHER_MESSAGE_MAX_BYTES`.
    """
    return _encode(LauncherRequest.model_validate(request.model_dump(), strict=True))


def decode_launcher_request(payload: bytes) -> LauncherRequest:
    """Decode one exact request frame received from the verified API peer.

    Parameters
    ----------
    payload : bytes
        Complete frame payload.

    Returns
    -------
    LauncherRequest
        Strictly typed request; not an authorisation of the job itself.

    Raises
    ------
    ValueError
        Size, encoding, duplicate names, unknown fields, version or any field
        shape is invalid.
    """
    return LauncherRequest.model_validate_json(_checked_text(payload), strict=True)


def encode_launcher_response(response: LauncherResponse) -> bytes:
    """Serialise a validated response as canonical bounded JSON.

    Parameters
    ----------
    response : LauncherResponse
        Already validated response.

    Returns
    -------
    bytes
        Sorted, compact UTF-8 JSON within :data:`LAUNCHER_MESSAGE_MAX_BYTES`.
    """
    return _encode(LauncherResponse.model_validate(response.model_dump(), strict=True))


def decode_launcher_response(payload: bytes, *, request: LauncherRequest) -> LauncherResponse:
    """Decode a response and require exact correlation with the sent request.

    Parameters
    ----------
    payload : bytes
        Complete frame payload from the verified launcher peer.
    request : LauncherRequest
        The request this response must answer.

    Returns
    -------
    LauncherResponse
        Correlated launcher observation.

    Raises
    ------
    ValueError
        The frame is malformed, inconsistent, or answers a different request,
        operation, job or generation.
    """
    response = LauncherResponse.model_validate_json(_checked_text(payload), strict=True)
    if (
        response.request_id != request.request_id
        or response.operation != request.operation
        or response.job_id != request.job_id
        or response.generation != request.generation
    ):
        raise ValueError("launcher response does not answer the request")
    return response
