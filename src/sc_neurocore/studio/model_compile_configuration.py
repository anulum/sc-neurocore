# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Canonical selected-model compiler configuration

"""Resolve one Studio catalogue selection into the canonical RTL compiler input."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema
from sc_neurocore.studio.model_catalogue import get_model_detail

ModelDetailGetter = Callable[[str], dict[str, object] | None]


@dataclass(frozen=True, slots=True)
class ResolvedModelCompileConfiguration:
    """Validated schema, compiler options and instantiated universal neuron."""

    dt: float
    integrator: str
    model_name: str
    module_name: str
    neuron: UniversalNeuron
    params: dict[str, float]
    q_format: QFormat
    schema_name: str
    schema_sha256: str

    def to_public_dict(self) -> dict[str, object]:
        """Return the path-free configuration attached to Studio evidence."""

        return {
            "dt": self.dt,
            "integrator": self.integrator,
            "model_name": self.model_name,
            "q_format": self.q_format.q_label,
            "schema_name": self.schema_name,
            "schema_sha256": self.schema_sha256,
        }

    def to_verilog(self) -> str:
        """Compile the resolved neuron with the exact selected fixed-point geometry."""

        return self.neuron.to_verilog(
            module_name=self.module_name,
            data_width=self.q_format.total_bits,
            fraction=self.q_format.fraction_bits,
        )


def resolve_model_compile_configuration(
    payload: Mapping[str, object],
    *,
    detail_getter: ModelDetailGetter = get_model_detail,
) -> ResolvedModelCompileConfiguration:
    """Validate a model-mode compiler payload and instantiate its canonical schema."""

    model_name = _required_string(payload, "model_name")
    params = _float_mapping(payload.get("params"))
    requested_dt = _optional_positive_float(payload.get("dt"), "dt")
    requested_integrator = _optional_string(payload.get("integrator"), "integrator")
    q_format = _q_format(_required_string(payload, "q_format", default="Q8.8"))
    requested_module_name = _optional_string(payload.get("module_name"), "module_name")

    detail = detail_getter(model_name)
    if detail is None:
        raise ValueError(f"Unknown Studio model {model_name!r}.")
    configuration = detail.get("compile_configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"Studio model {model_name!r} has no canonical schema.")

    schema_name = _required_string(configuration, "schema_name")
    schema = load_schema(schema_name)
    schema_sha256 = hashlib.sha256(
        json.dumps(schema, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    integration = schema.get("integration", {})
    if not isinstance(integration, Mapping):
        raise ValueError(
            f"Studio model compile configuration for {model_name!r} has invalid integration."
        )
    default_integrator = str(integration.get("method", "euler"))
    allowed_integrators = _string_list(configuration, "integrators")
    integrator = requested_integrator or default_integrator
    if integrator not in allowed_integrators:
        raise ValueError(f"Integrator {integrator!r} is not declared for {model_name!r}.")

    schema_params = schema.get("parameters", {})
    if not isinstance(schema_params, Mapping):
        raise ValueError(
            f"Studio model compile configuration for {model_name!r} has invalid parameters."
        )
    unknown_params = sorted(set(params) - set(schema_params))
    if unknown_params:
        raise ValueError(f"Unknown schema parameter override(s): {', '.join(unknown_params)}")

    dt = requested_dt
    if dt is None:
        dt = _positive_float(integration.get("dt", detail.get("dt")), "schema dt")
    module_name = requested_module_name or f"sc_{schema_name}_neuron"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module_name) is None:
        raise ValueError(
            "Studio model configuration field 'module_name' must be a Verilog identifier."
        )
    neuron = UniversalNeuron.from_schema(
        schema_name,
        parameter_overrides=params,
        dt_override=dt,
        method_override=integrator,
    )
    return ResolvedModelCompileConfiguration(
        dt=dt,
        integrator=integrator,
        model_name=model_name,
        module_name=module_name,
        neuron=neuron,
        params=params,
        q_format=q_format,
        schema_name=schema_name,
        schema_sha256=schema_sha256,
    )


def _q_format(value: str) -> QFormat:
    q_format = QFormat.from_string(value)
    if not 2 <= q_format.total_bits <= 64 or q_format.fraction_bits >= q_format.total_bits:
        raise ValueError("Studio RTL Q-format must be signed and between 2 and 64 total bits.")
    return q_format


def _required_string(payload: Mapping[str, object], key: str, *, default: str | None = None) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Studio model configuration field {key!r} must be a non-empty string.")
    return value


def _optional_string(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Studio model configuration field {key!r} must be a string or null.")
    return value


def _float_mapping(value: object) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Studio model configuration field 'params' must be an object.")
    result: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                "Studio model configuration parameter overrides must be finite numbers."
            )
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError(
                "Studio model configuration parameter overrides must be finite numbers."
            )
        result[key] = numeric
    return result


def _optional_positive_float(value: object, key: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, key)


def _positive_float(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Studio model configuration field {key!r} must be a positive number.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"Studio model configuration field {key!r} must be a positive number.")
    return numeric


def _string_list(payload: Mapping[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Studio model compile configuration {key!r} is invalid.")
    return value
