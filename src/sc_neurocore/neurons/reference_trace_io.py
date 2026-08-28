# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuron reference-trace corpus loading

"""Load and validate committed neuron reference-trace corpus entries."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
import json
import math
import sys
from types import MappingProxyType
from typing import cast

if sys.version_info >= (3, 11):
    from importlib.resources.abc import Traversable
else:  # pragma: no cover - Python 3.10 exposes Traversable via importlib.abc
    from importlib.abc import Traversable

from sc_neurocore.neurons.reference_trace_contracts import (
    REFERENCE_TRACE_SCHEMA_VERSION,
    FeatureTolerance,
    ReferenceTraceProtocol,
    ReferenceTraceProvenance,
    ReferenceTraceSpec,
)
from sc_neurocore.neurons.universal_dsl import list_bundled_schemas

_UNIVERSAL_DSL_REFERENCE_RUNNER = "universal_dsl"
_RUNNER_USES_DEDICATED_VALIDATOR: Mapping[str, bool] = MappingProxyType(
    {
        _UNIVERSAL_DSL_REFERENCE_RUNNER: False,
        "hand_model": True,
        "hand_and_universal_dsl": True,
    }
)


def reference_trace_spec_from_payload(payload: Mapping[str, object]) -> ReferenceTraceSpec:
    """Parse and validate a reference-trace corpus payload.

    Parameters
    ----------
    payload:
        JSON-compatible mapping using schema
        ``sc-neurocore.reference-trace.v1``.

    Returns
    -------
    ReferenceTraceSpec
        Validated immutable reference-trace specification.

    Raises
    ------
    ValueError
        If any required field is missing, malformed, non-finite, or references
        an unsupported runner or bundled neuron schema.
    """
    schema_version = _string_field(payload, "schema_version")
    if schema_version != REFERENCE_TRACE_SCHEMA_VERSION:
        raise ValueError(
            "reference trace schema_version must be "
            f"{REFERENCE_TRACE_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    model = _mapping_field(payload, "model")
    protocol_payload = _mapping_field(payload, "protocol")
    provenance_payload = _mapping_field(payload, "provenance")

    name = _string_field(payload, "name")
    schema_name = _string_field(model, "schema_name")
    runner = _string_field(model, "runner")
    if runner != _UNIVERSAL_DSL_REFERENCE_RUNNER:
        raise ValueError(f"reference trace runner {runner!r} is not supported")
    if schema_name not in set(list_bundled_schemas()):
        raise ValueError(f"reference trace schema {schema_name!r} is not bundled")

    expected_features = _float_mapping_field(payload, "expected_features")
    tolerances = _tolerances_from_payload(_mapping_field(payload, "tolerances"), expected_features)

    return ReferenceTraceSpec(
        name=name,
        schema_name=schema_name,
        runner=runner,
        protocol=ReferenceTraceProtocol(
            dt=_positive_float_field(protocol_payload, "dt"),
            steps=_positive_int_field(protocol_payload, "steps"),
            inputs=_float_mapping_field(protocol_payload, "inputs"),
            state_variables=_string_tuple_field(protocol_payload, "state_variables"),
            parameter_overrides=_optional_float_mapping_field(
                protocol_payload, "parameter_overrides"
            ),
        ),
        provenance=ReferenceTraceProvenance(
            kind=_string_field(provenance_payload, "kind"),
            source=_string_field(provenance_payload, "source"),
            equation=_string_field(provenance_payload, "equation"),
            citation=_optional_string_field(provenance_payload, "citation"),
        ),
        expected_features=expected_features,
        tolerances=tolerances,
    )


def list_reference_trace_specs() -> tuple[str, ...]:
    """Return all committed deterministic Universal-DSL trace names.

    Returns
    -------
    tuple[str, ...]
        Sorted stable identifiers for every deterministic scalar-feature
        payload in the corpus. Hand-model and seeded statistical artefacts use
        dedicated validators, so they are not coerced into this trace contract.
    """
    return tuple(spec.name for spec in _load_all_specs())


def load_reference_trace_spec(name: str) -> ReferenceTraceSpec:
    """Load one committed reference-trace specification.

    Parameters
    ----------
    name:
        Stable corpus identifier.

    Returns
    -------
    ReferenceTraceSpec
        Validated reference-trace specification.

    Raises
    ------
    ValueError
        If ``name`` is not present in the committed corpus.
    """
    for spec in _load_all_specs():
        if spec.name == name:
            return spec
    raise ValueError(f"unknown reference trace {name!r}")


def _load_all_specs() -> tuple[ReferenceTraceSpec, ...]:
    files = sorted(
        (
            path
            for path in _reference_trace_data_dir().iterdir()
            if path.is_file() and path.name.endswith(".json")
        ),
        key=lambda path: path.name,
    )
    loaded = (_load_spec_file(path) for path in files)
    return tuple(spec for spec in loaded if spec is not None)


def _load_spec_file(path: Traversable) -> ReferenceTraceSpec | None:
    """Load one corpus file, or skip non-universal-reference payloads.

    The directory may also hold dedicated identity/spec artefacts that use a
    different schema (for example map primary specs without a ``model`` block).
    Those files are intentionally skipped rather than treated as hard failures
    so the universal reference-trace loader stays fail-closed only for the
    ``sc-neurocore.reference-trace.v1`` corpus.
    """
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    mapping = _mapping_value(payload, path.name)
    raw_model = mapping.get("model")
    if isinstance(raw_model, Mapping) and "runner" in raw_model:
        runner = _string_field(cast(Mapping[str, object], raw_model), "runner")
        if runner not in _RUNNER_USES_DEDICATED_VALIDATOR:
            raise ValueError(f"reference trace runner {runner!r} is not supported")
    schema_version = mapping.get("schema_version")
    if schema_version != REFERENCE_TRACE_SCHEMA_VERSION:
        return None
    if "model" not in mapping:
        return None
    model = _mapping_field(mapping, "model")
    runner = _string_field(model, "runner")
    uses_dedicated_validator = _RUNNER_USES_DEDICATED_VALIDATOR[runner]
    if uses_dedicated_validator:
        return None
    return reference_trace_spec_from_payload(mapping)


def _reference_trace_data_dir() -> Traversable:
    return resources.files("sc_neurocore.neurons").joinpath("reference_trace_data")


def _field(payload: Mapping[str, object], key: str) -> object:
    if key not in payload:
        raise ValueError(f"reference trace payload is missing {key!r}")
    return payload[key]


def _string_field(payload: Mapping[str, object], key: str) -> str:
    value = _field(payload, key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"reference trace field {key!r} must be a non-empty string")
    return value


def _optional_string_field(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise ValueError(f"reference trace field {key!r} must be a non-empty string when set")
    return value


def _mapping_field(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = _field(payload, key)
    if not isinstance(value, Mapping):
        raise ValueError(f"reference trace field {key!r} must be an object")
    if not all(isinstance(item_key, str) for item_key in value):
        raise ValueError(f"reference trace object {key!r} must use string keys")
    return cast(Mapping[str, object], value)


def _float_field(payload: Mapping[str, object], key: str) -> float:
    value = _field(payload, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"reference trace field {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"reference trace field {key!r} must be finite")
    return result


def _positive_float_field(payload: Mapping[str, object], key: str) -> float:
    value = _float_field(payload, key)
    if value <= 0.0:
        raise ValueError(f"reference trace field {key!r} must be positive")
    return value


def _positive_int_field(payload: Mapping[str, object], key: str) -> int:
    value = _field(payload, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"reference trace field {key!r} must be an integer")
    if value <= 0:
        raise ValueError(f"reference trace field {key!r} must be positive")
    return value


def _float_mapping_field(payload: Mapping[str, object], key: str) -> Mapping[str, float]:
    mapping = _mapping_field(payload, key)
    values: dict[str, float] = {}
    for item_key, value in mapping.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"reference trace field {key!r}.{item_key} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"reference trace field {key!r}.{item_key} must be finite")
        values[item_key] = result
    if not values:
        raise ValueError(f"reference trace field {key!r} must not be empty")
    return MappingProxyType(values)


def _optional_float_mapping_field(payload: Mapping[str, object], key: str) -> Mapping[str, float]:
    if key not in payload:
        return MappingProxyType({})
    mapping = _mapping_field(payload, key)
    values: dict[str, float] = {}
    for item_key, value in mapping.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"reference trace field {key!r}.{item_key} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"reference trace field {key!r}.{item_key} must be finite")
        values[item_key] = result
    return MappingProxyType(values)


def _string_tuple_field(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = _field(payload, key)
    if not isinstance(value, list):
        raise ValueError(f"reference trace field {key!r} must be a list")
    result = tuple(item for item in value if isinstance(item, str) and item != "")
    if len(result) != len(value):
        raise ValueError(f"reference trace field {key!r} must contain non-empty strings")
    return result


def _tolerances_from_payload(
    payload: Mapping[str, object], features: Mapping[str, float]
) -> Mapping[str, FeatureTolerance]:
    default_payload = payload.get("default")
    default_tolerance = (
        _tolerance_from_payload("default", _mapping_value(default_payload, "tolerances.default"))
        if default_payload is not None
        else None
    )
    result: dict[str, FeatureTolerance] = {}
    for feature in features:
        feature_payload = payload.get(feature)
        if feature_payload is None:
            if default_tolerance is None:
                raise ValueError(f"reference trace tolerance missing for feature {feature!r}")
            result[feature] = default_tolerance
        else:
            result[feature] = _tolerance_from_payload(
                feature, _mapping_value(feature_payload, f"tolerances.{feature}")
            )
    return MappingProxyType(result)


def _mapping_value(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"reference trace field {label!r} must be an object")
    if not all(isinstance(item_key, str) for item_key in value):
        raise ValueError(f"reference trace object {label!r} must use string keys")
    return cast(Mapping[str, object], value)


def _tolerance_from_payload(label: str, payload: Mapping[str, object]) -> FeatureTolerance:
    absolute = _float_field(payload, "absolute")
    relative_value = payload.get("relative", 0.0)
    if isinstance(relative_value, bool) or not isinstance(relative_value, int | float):
        raise ValueError(f"reference trace tolerance {label!r} relative value must be numeric")
    relative = float(relative_value)
    if not math.isfinite(relative):
        raise ValueError(f"reference trace tolerance {label!r} relative value must be finite")
    if absolute < 0.0 or relative < 0.0:
        raise ValueError(f"reference trace tolerance {label!r} values must be non-negative")
    return FeatureTolerance(absolute=absolute, relative=relative)


__all__ = [
    "list_reference_trace_specs",
    "load_reference_trace_spec",
    "reference_trace_spec_from_payload",
]
