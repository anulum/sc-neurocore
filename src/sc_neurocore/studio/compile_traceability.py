# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio compile traceability

"""Path-free source-to-RTL traceability contracts for Studio compile results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

from sc_neurocore.studio.evidence_classification import (
    StudioEvidenceClassification,
    StudioEvidenceStatus,
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)

STUDIO_COMPILE_TRACEABILITY_SCHEMA_VERSION = "studio.compile-traceability.v1"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FloatMap: TypeAlias = Mapping[str, float]


@dataclass(frozen=True, slots=True)
class StudioCompileTraceability:
    """Path-free source-to-RTL provenance for a Studio compile result.

    Parameters
    ----------
    equations:
        ODE equation strings submitted by the operator.
    threshold:
        Optional spike-threshold expression.
    reset:
        Optional reset expression.
    params:
        Numeric parameter overrides used by the compiler.
    init:
        Initial state values supplied with the request.
    module_name:
        RTL module name requested by the operator.
    verilog:
        Generated Verilog/SystemVerilog source.
    source:
        Stable source-kind label for the compile request.
    output_language:
        RTL language emitted by the backend compiler.
    evidence_classification:
        Evidence lane label consumed by Studio evidence bundles.
    status:
        Terminal status for this compile traceability object.
    """

    equations: tuple[str, ...]
    threshold: str | None
    reset: str | None
    params: dict[str, float]
    init: dict[str, float]
    module_name: str
    verilog: str
    source: str = "ode"
    output_language: str = "verilog"
    evidence_classification: StudioEvidenceClassification = "compile"
    status: StudioEvidenceStatus = "completed"

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the public, path-free traceability payload."""

        source_payload: dict[str, JsonValue] = {
            "equations": list(self.equations),
            "init": cast(dict[str, JsonValue], self.init),
            "params": cast(dict[str, JsonValue], self.params),
            "reset": self.reset,
            "threshold": self.threshold,
        }
        output_payload: dict[str, JsonValue] = {
            "language": self.output_language,
            "module_name": self.module_name,
            "rtl_chars": len(self.verilog),
            "rtl_sha256": _sha256_text(self.verilog),
        }
        payload: dict[str, JsonValue] = {
            "evidence_classification": validate_studio_evidence_classification(
                self.evidence_classification
            ),
            "input_sha256": _sha256_json(source_payload),
            "output": output_payload,
            "schema_version": STUDIO_COMPILE_TRACEABILITY_SCHEMA_VERSION,
            "source": self.source,
            "source_payload": source_payload,
            "status": validate_studio_evidence_status(self.status),
        }
        payload["traceability_sha256"] = _sha256_json(payload)
        return payload


def build_compile_traceability(
    *,
    equations: Sequence[str],
    threshold: str | None,
    reset: str | None,
    params: FloatMap | None,
    init: FloatMap | None,
    module_name: str,
    verilog: str,
) -> StudioCompileTraceability:
    """Build path-free traceability for an equation-to-RTL compile result.

    Parameters
    ----------
    equations:
        ODE equation strings submitted by the operator.
    threshold:
        Optional spike-threshold expression.
    reset:
        Optional reset expression.
    params:
        Numeric parameter overrides used by the compiler.
    init:
        Initial state values supplied with the request.
    module_name:
        RTL module name requested by the operator.
    verilog:
        Generated Verilog/SystemVerilog source.

    Returns
    -------
    StudioCompileTraceability
        Immutable traceability record with stable public JSON conversion.

    Raises
    ------
    ValueError
        If no equations are supplied.
    """

    if not equations:
        raise ValueError("At least one equation is required for compile traceability.")
    return StudioCompileTraceability(
        equations=tuple(equations),
        threshold=threshold,
        reset=reset,
        params=dict(params or {}),
        init=dict(init or {}),
        module_name=module_name,
        verilog=verilog,
    )


def _sha256_text(value: str) -> str:
    """Return a SHA-256 digest for UTF-8 text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(payload: Mapping[str, JsonValue]) -> str:
    """Return a stable SHA-256 digest over canonical JSON."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
