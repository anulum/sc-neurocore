# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio compiler process task

"""Importable process task for the Studio compiler endpoint."""

from __future__ import annotations

import json
from collections.abc import Mapping

from sc_neurocore.compiler.equation_compiler import equation_to_fpga
from sc_neurocore.studio.compile_traceability import build_compile_traceability
from sc_neurocore.studio.platform.action_evidence import (
    write_studio_action_evidence_manifest,
)
from sc_neurocore.studio.platform.jobs import StudioJobContext

COMPILE_PROCESS_TASK = "sc_neurocore.studio.platform.compile_process:run_compile_process_task"


def run_compile_process_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Compile one Studio ODE payload in an isolated process.

    Parameters
    ----------
    context:
        Job sandbox used to write path-confined result and evidence artifacts.
    payload:
        JSON object matching the public ``/api/compile`` request contract.

    Returns
    -------
    dict[str, object]
        Path-free compile response payload.

    Raises
    ------
    ValueError
        If the payload does not match the JSON-serializable compile contract.
    """

    request = _compile_request_from_payload(payload)
    _, verilog = equation_to_fpga(
        request.equations[0],
        threshold=request.threshold,
        reset=request.reset,
        params=request.params,
        init=request.init,
        module_name=request.module_name,
    )
    result: dict[str, object] = {
        "chars": len(verilog),
        "compile_traceability": build_compile_traceability(
            equations=request.equations,
            threshold=request.threshold,
            reset=request.reset,
            params=request.params,
            init=request.init,
            module_name=request.module_name,
            verilog=verilog,
        ).to_public_dict(),
        "module_name": request.module_name,
        "verilog": verilog,
    }
    result_artifact = context.write_artifact(
        "compiler/result.json",
        _serialize_worker_result(result),
    )
    write_studio_action_evidence_manifest(
        context,
        action_kind="studio.compile",
        result=result,
        result_artifact=result_artifact,
        evidence_artifact_path="compiler/evidence.json",
        evidence_classification="compile",
        replay_route="POST /api/compile",
    )
    return result


class _CompileProcessRequest:
    """Validated JSON payload for an isolated compiler task."""

    def __init__(
        self,
        *,
        equations: list[str],
        threshold: str | None,
        reset: str | None,
        params: dict[str, float] | None,
        init: dict[str, float] | None,
        module_name: str,
    ) -> None:
        self.equations = equations
        self.threshold = threshold
        self.reset = reset
        self.params = params
        self.init = init
        self.module_name = module_name


def _compile_request_from_payload(payload: Mapping[str, object]) -> _CompileProcessRequest:
    return _CompileProcessRequest(
        equations=_string_list_field(payload, "equations"),
        threshold=_optional_string_field(payload, "threshold"),
        reset=_optional_string_field(payload, "reset"),
        params=_optional_float_mapping_field(payload, "params"),
        init=_optional_float_mapping_field(payload, "init"),
        module_name=_string_field(payload, "module_name", default="sc_neuron"),
    )


def _string_list_field(payload: Mapping[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Studio compile payload field {key!r} must be a non-empty list.")
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Studio compile payload field {key!r} must contain strings.")
        strings.append(item)
    return strings


def _optional_string_field(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Studio compile payload field {key!r} must be a string or null.")
    return value


def _string_field(payload: Mapping[str, object], key: str, *, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Studio compile payload field {key!r} must be a non-empty string.")
    return value


def _optional_float_mapping_field(
    payload: Mapping[str, object],
    key: str,
) -> dict[str, float] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Studio compile payload field {key!r} must be an object or null.")
    parsed: dict[str, float] = {}
    for raw_name, raw_value in value.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"Studio compile payload field {key!r} contains an invalid name.")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError(f"Studio compile payload field {key!r} contains a non-numeric value.")
        parsed[raw_name] = float(raw_value)
    return parsed


def _serialize_worker_result(result: Mapping[str, object]) -> str:
    return json.dumps(dict(result), sort_keys=True, default=str)


__all__ = [
    "COMPILE_PROCESS_TASK",
    "run_compile_process_task",
]
