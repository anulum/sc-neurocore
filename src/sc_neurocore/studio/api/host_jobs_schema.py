# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — closed hosted job requests

"""Validate hosted operations without rewriting the original scientific JSON."""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from sc_neurocore.studio.api.analysis_jobs import (
    AnalysisKind,
    validate_analysis_job_request,
)
from sc_neurocore.studio.api.schemas import AnalysisJobRequest
from sc_neurocore.studio.equation_ir_schema import (
    EquationIREmitRequest,
    EquationIRRequest,
    EquationIRTextRequest,
)

HostCompilerOperation = Literal["ir.build", "ir.verify", "ir.replay", "ir.emit-sv"]
HostOperation = AnalysisKind | HostCompilerOperation
HOST_ANALYSIS_OPERATIONS: tuple[AnalysisKind, ...] = (
    "simulate",
    "fi_curve",
    "bifurcation",
    "heatmap",
    "sensitivity",
)
HOST_COMPILER_OPERATIONS: tuple[HostCompilerOperation, ...] = (
    "ir.build",
    "ir.verify",
    "ir.replay",
    "ir.emit-sv",
)
HOST_REQUEST_MAX_BYTES = 1_048_576


class HostBridgeDescription(BaseModel):
    """Installed local operation families under the closed host wire contract.

    Attributes
    ----------
    protocol, profile : str
        Fixed protocol and local-first execution boundary, without identity grants.
    operations : tuple of str
        Known actions whose required dependency families imported successfully.
        This is availability, not scientific or hardware qualification.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    protocol: Literal["sc-neurocore.host-api.v1"] = "sc-neurocore.host-api.v1"
    profile: Literal["local-first"] = "local-first"
    operations: tuple[HostOperation, ...]


class HostAnalysisRequest(BaseModel):
    """One existing simulation or analysis request selected by a closed operation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    operation: AnalysisKind
    request: dict[str, JsonValue]


class HostIRBuildRequest(BaseModel):
    """An ordered authored equation model and its explicit compiler profile."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    operation: Literal["ir.build"]
    request: EquationIRRequest


class HostIRTextRequest(BaseModel):
    """A versioned native graph for structural verification or source replay."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    operation: Literal["ir.verify", "ir.replay"]
    request: EquationIRTextRequest


class HostIREmitRequest(BaseModel):
    """A versioned native graph and explicit emission source policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    operation: Literal["ir.emit-sv"]
    request: EquationIREmitRequest


HostJobRequest = HostAnalysisRequest | HostIRBuildRequest | HostIRTextRequest | HostIREmitRequest
_REQUEST: TypeAdapter[HostJobRequest] = TypeAdapter(
    Annotated[HostJobRequest, Field(discriminator="operation")]
)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse ambiguous names at every nesting level before typed validation."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate hosted request field")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """Refuse nonfinite extensions to the scientific JSON contract."""
    raise ValueError("Nonfinite hosted request value")


def _finite_float(value: str) -> float:
    """Reject overflowed JSON coefficients before the scientific schema sees them."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite hosted request value")
    return number


def parse_host_job(text: str) -> HostJobRequest:
    """Validate a complete bounded envelope without changing its stored text.

    Parameters
    ----------
    text : str
        Original UTF-8 HTTP object, including the nested scientific request.

    Returns
    -------
    HostJobRequest
        Closed typed operation with the existing scientific schema validated.

    Raises
    ------
    ValueError
        Encoding, size, JSON, operation or scientific input is invalid.
    """
    if not 0 < len(text.encode("utf-8")) <= HOST_REQUEST_MAX_BYTES:
        raise ValueError("Hosted request exceeds its byte limit")
    try:
        json.loads(
            text,
            object_pairs_hook=_unique_fields,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
        request = _REQUEST.validate_json(text)
        if isinstance(request, HostAnalysisRequest):
            validate_analysis_job_request(
                AnalysisJobRequest(analysis=request.operation, payload=request.request)
            )
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("Invalid hosted request JSON") from exc
    return request
