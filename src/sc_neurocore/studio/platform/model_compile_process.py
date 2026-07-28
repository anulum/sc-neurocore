# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio catalogue-model compiler process task

"""Compile one canonical catalogue schema in an isolated Studio job."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema
from sc_neurocore.studio.compile_traceability import build_model_compile_traceability
from sc_neurocore.studio.model_catalogue import get_model_detail
from sc_neurocore.studio.platform.action_evidence import write_studio_action_evidence_manifest
from sc_neurocore.studio.platform.jobs import StudioJobContext

MODEL_COMPILE_PROCESS_TASK = (
    "sc_neurocore.studio.platform.model_compile_process:run_model_compile_process_task"
)


@dataclass(frozen=True, slots=True)
class _ModelCompileRequest:
    model_name: str
    params: dict[str, float]
    dt: float | None
    integrator: str | None
    q_format: str
    module_name: str | None


def run_model_compile_process_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Resolve and compile one catalogue model through its canonical schema."""

    request = _request_from_payload(payload)
    detail = get_model_detail(request.model_name)
    if detail is None:
        raise ValueError(f"Unknown Studio model {request.model_name!r}.")
    configuration = detail.get("compile_configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"Studio model {request.model_name!r} has no canonical schema.")

    schema_name = _required_string(configuration, "schema_name")
    schema = load_schema(schema_name)
    schema_sha256 = hashlib.sha256(
        json.dumps(schema, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    integration = schema.get("integration", {})
    default_integrator = str(integration.get("method", "euler"))
    allowed_integrators = _string_list(configuration, "integrators")
    integrator = request.integrator or default_integrator
    if integrator not in allowed_integrators:
        raise ValueError(f"Integrator {integrator!r} is not declared for {request.model_name!r}.")

    schema_params = schema.get("parameters", {})
    unknown_params = sorted(set(request.params) - set(schema_params))
    if unknown_params:
        raise ValueError(f"Unknown schema parameter override(s): {', '.join(unknown_params)}")

    q_format = QFormat.from_string(request.q_format)
    if not 2 <= q_format.total_bits <= 64 or q_format.fraction_bits >= q_format.total_bits:
        raise ValueError("Studio RTL Q-format must be signed and between 2 and 64 total bits.")

    dt = request.dt
    if dt is None:
        dt = _positive_float(integration.get("dt", detail.get("dt")), "schema dt")
    module_name = request.module_name or f"sc_{schema_name}_neuron"
    neuron = UniversalNeuron.from_schema(
        schema_name,
        parameter_overrides=request.params,
        dt_override=dt,
        method_override=integrator,
    )
    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=q_format.total_bits,
        fraction=q_format.fraction_bits,
    )
    compile_configuration: dict[str, object] = {
        "dt": dt,
        "integrator": integrator,
        "model_name": request.model_name,
        "q_format": q_format.q_label,
        "schema_name": schema_name,
        "schema_sha256": schema_sha256,
    }
    result: dict[str, object] = {
        "chars": len(verilog),
        "compile_configuration": compile_configuration,
        "compile_traceability": build_model_compile_traceability(
            model_name=request.model_name,
            schema_name=schema_name,
            schema_sha256=schema_sha256,
            params=request.params,
            dt=dt,
            integrator=integrator,
            q_format=q_format.q_label,
            module_name=module_name,
            verilog=verilog,
        ).to_public_dict(),
        "module_name": module_name,
        "verilog": verilog,
    }
    result_artifact = context.write_artifact(
        "compiler/model-result.json",
        json.dumps(result, allow_nan=False, sort_keys=True).encode("utf-8"),
    )
    write_studio_action_evidence_manifest(
        context,
        action_kind="studio.models.compile",
        result=result,
        result_artifact=result_artifact,
        evidence_artifact_path="compiler/model-evidence.json",
        evidence_classification="compile",
        replay_route="POST /api/models/compile",
    )
    return result


def _request_from_payload(payload: Mapping[str, object]) -> _ModelCompileRequest:
    return _ModelCompileRequest(
        model_name=_required_string(payload, "model_name"),
        params=_float_mapping(payload.get("params")),
        dt=_optional_positive_float(payload.get("dt"), "dt"),
        integrator=_optional_string(payload.get("integrator"), "integrator"),
        q_format=_required_string(payload, "q_format", default="Q8.8"),
        module_name=_optional_string(payload.get("module_name"), "module_name"),
    )


def _required_string(payload: Mapping[str, object], key: str, *, default: str | None = None) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Studio model compile field {key!r} must be a non-empty string.")
    return value


def _optional_string(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Studio model compile field {key!r} must be a string or null.")
    return value


def _float_mapping(value: object) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Studio model compile field 'params' must be an object.")
    result: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("Studio model compile parameter overrides must be finite numbers.")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError("Studio model compile parameter overrides must be finite numbers.")
        result[key] = numeric
    return result


def _optional_positive_float(value: object, key: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, key)


def _positive_float(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Studio model compile field {key!r} must be a positive number.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"Studio model compile field {key!r} must be a positive number.")
    return numeric


def _string_list(payload: Mapping[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Studio model compile configuration {key!r} is invalid.")
    return value
