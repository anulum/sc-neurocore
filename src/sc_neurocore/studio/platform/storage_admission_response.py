# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — correlated storage admission outcome

"""Encode named-admission outcomes without mistaking transport for success."""

from __future__ import annotations

import json
from typing import Annotated, Literal, cast
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_admission_replay import StorageAdmissionReplay
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission
from sc_neurocore.studio.platform.storage_admission_protocol import StorageNamedAdmissionRequest
from sc_neurocore.studio.platform.storage_named_tasks import resolve_named_studio_task

_Name = Annotated[str, Field(min_length=1)]
_Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_Count = Annotated[int, Field(ge=0)]


class StorageNamedAdmissionResponse(BaseModel):
    """One exact admission or capacity-refusal outcome from the authority.

    A caller must supply the result of its durable authority transaction; the
    codec cannot prove persistence from a Python object alone. A lost reply
    remains ambiguous until the same mutation ID and content are replayed
    against the authority's durable table.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    schema_version: Literal["studio.storage.admission.v1"]
    operation: Literal["admit_named_result"]
    request_id: str | None
    mutation_id: _Name
    workspace: _Name
    task_name: _Name
    payload_sha256: _Digest
    status: Literal["admitted", "queue_full"]
    job_id: _Name | None
    running: _Count | None
    queued: _Count | None
    limit: _Count | None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Require mutually exclusive complete success and capacity shapes."""
        metrics = (self.running, self.queued, self.limit)
        if self.status == "admitted":
            if self.job_id is None or any(item is not None for item in metrics):
                raise ValueError("invalid admitted storage outcome")
        elif self.job_id is not None or any(item is None for item in metrics):
            raise ValueError("invalid queue-full storage outcome")
        return self


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON names at every response-object depth."""
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate storage admission response field")
        fields[name] = value
    return fields


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON extensions in an authority result."""
    raise ValueError("nonfinite storage admission response constant")


def _bound_request(request: StorageNamedAdmissionRequest, replay: StorageAdmissionReplay) -> None:
    """Require a validated replay identity for this exact named request."""
    if not isinstance(request, StorageNamedAdmissionRequest) or not isinstance(
        replay, StorageAdmissionReplay
    ):
        raise ValueError("invalid storage admission response correlation")
    replay.validate()
    claim = request.requester
    if (
        claim is None
        or replay.requester != claim.principal_id
        or replay.mutation_id != request.mutation_id
    ):
        raise ValueError("storage admission replay does not match request")
    resolve_named_studio_task(request.task_name, authorized_route=request.authorized_route)


def encode_named_admission_response(
    *,
    request: StorageNamedAdmissionRequest,
    replay: StorageAdmissionReplay,
    outcome: StudioJobSubmission | StudioJobQueueFull,
    max_bytes: int,
) -> bytes:
    """Serialize one actual authority outcome within a trusted frame ceiling.

    Parameters
    ----------
    request : StorageNamedAdmissionRequest
        Exact request already authorized and prepared by the service.
    replay : StorageAdmissionReplay
        Service-derived content identity used in the durable transaction.
    outcome : StudioJobSubmission or StudioJobQueueFull
        Admission or capacity refusal returned by the authority transaction.
    max_bytes : int
        Positive trusted maximum response-frame payload bytes.

    Returns
    -------
    bytes
        Strict versioned result to send over the peer-verified channel.

    Raises
    ------
    ValueError
        Correlation, domain outcome or complete frame size is invalid.
    """
    _bound_request(request, replay)
    if type(max_bytes) is not int or not 0 < max_bytes <= 0xFFFFFFFF:
        raise ValueError("invalid storage admission response limit")
    job_id: str | None = None
    running: int | None = None
    queued: int | None = None
    limit: int | None = None
    status: Literal["admitted", "queue_full"]
    if isinstance(outcome, StudioJobSubmission):
        task = resolve_named_studio_task(
            request.task_name, authorized_route=request.authorized_route
        )
        record = outcome.record
        if (
            record.workspace != request.workspace
            or record.kind != task.kind
            or record.owner != task.owner
            or record.execution_model != "process"
        ):
            raise ValueError("storage admission record does not match named request")
        job_id = record.job_id
        status = "admitted"
    elif isinstance(outcome, StudioJobQueueFull):
        status = "queue_full"
        running, queued, limit = outcome.running, outcome.queued, outcome.limit
    else:
        raise ValueError("unsupported storage admission outcome")
    response = StorageNamedAdmissionResponse(
        schema_version="studio.storage.admission.v1",
        operation="admit_named_result",
        request_id=request.request_id,
        mutation_id=request.mutation_id,
        workspace=request.workspace,
        task_name=request.task_name,
        payload_sha256=replay.payload_sha256,
        status=status,
        job_id=job_id,
        running=running,
        queued=queued,
        limit=limit,
    )
    encoded = response.model_dump_json().encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("storage admission response exceeds frame limit")
    return encoded


def decode_named_admission_response(
    payload: bytes,
    *,
    request: StorageNamedAdmissionRequest,
    replay: StorageAdmissionReplay,
    max_bytes: int,
) -> str:
    """Return the admitted job ID or raise the exact durable queue refusal.

    This codec checks a bounded response from a separately verified service
    peer. It cannot prove that the worker was launched or safely supervised;
    the service may emit success only after its own custody and admission gate.
    A lost reply is ambiguous and must use durable mutation replay, not an
    automatic fresh submission.

    Parameters
    ----------
    payload : bytes
        Complete response frame from a separately peer-verified service.
    request : StorageNamedAdmissionRequest
        Exact submitted metadata retained by the trusted API caller.
    replay : StorageAdmissionReplay
        Client-computed expected digest of the frozen request and seed bytes.
    max_bytes : int
        Positive trusted maximum response-frame payload bytes.

    Returns
    -------
    str
        Correlated admitted job ID; fetch its complete record separately.

    Raises
    ------
    StudioJobQueueFull
        A correlated durable capacity refusal.
    ValueError
        Framing, JSON, schema, outcome or request correlation is invalid.
    """
    _bound_request(request, replay)
    if (
        type(max_bytes) is not int
        or not 0 < max_bytes <= 0xFFFFFFFF
        or not isinstance(payload, bytes)
        or not 0 < len(payload) <= max_bytes
    ):
        raise ValueError("storage admission response exceeds frame limit")
    try:
        text = payload.decode("utf-8")
        json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid storage admission response JSON") from exc
    response = StorageNamedAdmissionResponse.model_validate_json(text, strict=True)
    if (
        response.request_id != request.request_id
        or response.mutation_id != request.mutation_id
        or response.workspace != request.workspace
        or response.task_name != request.task_name
        or response.payload_sha256 != replay.payload_sha256
    ):
        raise ValueError("storage admission response does not match request")
    if response.status == "queue_full":
        # The response model's validator proved every metric is present.
        raise StudioJobQueueFull(
            running=cast(int, response.running),
            queued=cast(int, response.queued),
            limit=cast(int, response.limit),
        )
    return cast(str, response.job_id)
