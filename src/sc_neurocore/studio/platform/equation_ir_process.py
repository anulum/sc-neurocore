# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded authored equation compiler tasks

"""Run native equation actions through real process admission and evidence storage."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from typing_extensions import Self

from sc_neurocore.studio.equation_ir import (
    EquationIRRefusal,
    EquationIRResult,
    IRRefusalCode,
    IRStage,
    build_equation_ir,
    emit_equation_systemverilog,
    replay_equation_ir,
    verify_equation_ir,
)
from sc_neurocore.studio.equation_ir_schema import (
    EquationIREmitRequest,
    EquationIRRequest,
    EquationIRTextRequest,
)
from sc_neurocore.studio.platform.action_evidence import (
    write_studio_action_evidence_manifest,
)
from sc_neurocore.studio.platform.jobs import StudioJobContext

EQUATION_IR_PROCESS_TASK = (
    "sc_neurocore.studio.platform.equation_ir_process:run_equation_ir_process_task"
)


class EquationIRFailure(BaseModel):
    """Controlled refusal transported independently of the worker job status.

    Attributes
    ----------
    stage, code, message : str
        Authored compiler boundary, stable reason and caller-facing explanation.
    status_code : int
        Input refusal or unavailable dependency; never a success status.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    stage: IRStage
    code: IRRefusalCode
    message: str
    status_code: Literal[422, 503]

    @model_validator(mode="after")
    def _status(self) -> Self:
        """Prevent dependency failures from being mislabeled as input refusals."""
        expected = 503 if self.code == "dependency_unavailable" else 422
        if self.status_code != expected:
            raise ValueError("Compiler refusal status does not match its reason")
        return self


class EquationIRWorkerSuccess(BaseModel):
    """Accepted worker envelope with a required graph and no refusal.

    Attributes
    ----------
    accepted : bool
        Always true for this envelope.
    result : EquationIRResult
        Validated graph and compiler evidence.
    refusal : None
        A successful operation has no controlled refusal.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    accepted: Literal[True]
    result: EquationIRResult
    refusal: None


class EquationIRWorkerRefusal(BaseModel):
    """Refused worker envelope with a required reason and no graph.

    Attributes
    ----------
    accepted : bool
        Always false for this envelope.
    result : None
        A refused operation has no compiler graph result.
    refusal : EquationIRFailure
        Validated reason and corresponding response status.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    accepted: Literal[False]
    result: None
    refusal: EquationIRFailure


_WorkerOutcome = Annotated[
    EquationIRWorkerSuccess | EquationIRWorkerRefusal, Field(discriminator="accepted")
]


class EquationIRWorkerResult(RootModel[_WorkerOutcome]):
    """Validate the serialized success/refusal union without boolean coercion.

    Attributes
    ----------
    root : EquationIRWorkerSuccess or EquationIRWorkerRefusal
        One complete outcome; inconsistent optional field combinations cannot
        represent a validated worker result.
    """

    @model_validator(mode="before")
    @classmethod
    def _strict_accepted(cls, value: object) -> object:
        """Reject integer discriminator aliases admitted by Python bool literals."""
        if isinstance(value, Mapping) and type(value.get("accepted")) is not bool:
            raise ValueError("Compiler worker accepted must be a boolean")
        return value


class _BuildPayload(BaseModel):
    """Admit an exact ordered source request after worker serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal["build"]
    request: EquationIRRequest


class _TextPayload(BaseModel):
    """Admit structural verification or source replay after transport."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal["verify", "replay"]
    request: EquationIRTextRequest


class _EmitPayload(BaseModel):
    """Admit native emission with an explicit source requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal["emit-sv"]
    request: EquationIREmitRequest


_Payload = Annotated[_BuildPayload | _TextPayload | _EmitPayload, Field(discriminator="operation")]
_PAYLOAD_ADAPTER: TypeAdapter[_Payload] = TypeAdapter(_Payload)


def run_equation_ir_process_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Run one bounded native action and record its actual compiler outcome.

    Parameters
    ----------
    context : StudioJobContext
        Admitted worker sandbox and artifact writer.
    payload : Mapping of str to object
        Discriminated operation and versioned request from the public caller.

    Returns
    -------
    dict of str to object
        Successful graph result or controlled refusal. A completed worker is not
        necessarily a successful compilation; accepted and evidence status carry
        the compiler outcome separately from the job transport status.

    Notes
    -----
    Unexpected exceptions propagate to the supervised worker's generic failure
    boundary. They never become successful graph results or public diagnostics.
    """
    try:
        try:
            request = _PAYLOAD_ADAPTER.validate_python(dict(payload))
        except ValidationError as exc:
            raise EquationIRRefusal(
                "request", "invalid_input", "The compiler job request is invalid."
            ) from exc
        if isinstance(request, _BuildPayload):
            result = build_equation_ir(request.request)
        elif isinstance(request, _EmitPayload):
            result = emit_equation_systemverilog(request.request)
        elif request.operation == "verify":
            result = verify_equation_ir(request.request)
        else:
            result = replay_equation_ir(request.request)
    except EquationIRRefusal as exc:
        response: dict[str, object] = EquationIRWorkerRefusal(
            accepted=False,
            result=None,
            refusal=EquationIRFailure(
                stage=exc.stage,
                code=exc.code,
                message=exc.message,
                status_code=503 if exc.code == "dependency_unavailable" else 422,
            ),
        ).model_dump(mode="json")
    else:
        response = EquationIRWorkerSuccess(accepted=True, result=result, refusal=None).model_dump(
            mode="json"
        )

    artifact = context.write_artifact(
        "equation-ir/result.json",
        json.dumps(response, sort_keys=True, allow_nan=False),
    )
    write_studio_action_evidence_manifest(
        context,
        action_kind="studio.equation_ir",
        result=response,
        result_artifact=artifact,
        evidence_artifact_path="equation-ir/evidence.json",
        evidence_classification="compile",
        replay_route=_replay_route(payload),
        status="completed" if response["accepted"] else "failed",
    )
    return response


def _replay_route(payload: Mapping[str, object]) -> str:
    """Select an authored fixed route without interpolating arbitrary job input."""
    operation = payload.get("operation")
    if operation == "build":
        return "POST /api/ir/build"
    if operation == "verify":
        return "POST /api/ir/verify"
    if operation == "replay":
        return "POST /api/ir/replay"
    if operation == "emit-sv":
        return "POST /api/ir/emit-sv"
    raise ValueError("Compiler operation has no replay route")
