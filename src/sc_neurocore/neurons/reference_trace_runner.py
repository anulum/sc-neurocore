# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuron reference-trace execution

"""Execute and compare neuron reference-trace specifications."""

from __future__ import annotations

from collections.abc import Mapping
import math
from types import MappingProxyType

from sc_neurocore.neurons.reference_trace_contracts import (
    FeatureMismatch,
    ReferenceTraceSpec,
    TraceSimulationResult,
    TraceValidationReport,
)
from sc_neurocore.neurons.reference_trace_io import (
    list_reference_trace_specs,
    load_reference_trace_spec,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron


def simulate_reference_trace(spec_or_name: ReferenceTraceSpec | str) -> TraceSimulationResult:
    """Run a reference-trace protocol through ``UniversalNeuron``.

    Parameters
    ----------
    spec_or_name:
        Reference-trace specification or committed corpus name.

    Returns
    -------
    TraceSimulationResult
        Recorded state trace, spike sequence, and extracted feature map.
    """
    spec = _coerce_spec(spec_or_name)
    if spec.runner != "universal_dsl":
        raise ValueError(f"unsupported reference-trace runner {spec.runner!r}")

    neuron = UniversalNeuron.from_schema(
        spec.schema_name,
        dt_override=spec.protocol.dt,
        parameter_overrides=dict(spec.protocol.parameter_overrides),
    )
    recorded: dict[str, list[float]] = {name: [] for name in spec.protocol.state_variables}
    spikes: list[int] = []
    for _ in range(spec.protocol.steps):
        spikes.append(neuron.step(**dict(spec.protocol.inputs)))
        for variable in spec.protocol.state_variables:
            recorded[variable].append(float(neuron.state[variable]))

    trace = MappingProxyType({variable: tuple(values) for variable, values in recorded.items()})
    spike_tuple = tuple(spikes)
    return TraceSimulationResult(
        name=spec.name,
        steps=spec.protocol.steps,
        trace=trace,
        spikes=spike_tuple,
        features=_extract_features(trace, spike_tuple),
    )


def validate_reference_trace(name: str) -> TraceValidationReport:
    """Validate one committed reference trace by name.

    Parameters
    ----------
    name:
        Stable corpus identifier.

    Returns
    -------
    TraceValidationReport
        Feature-level validation report.
    """
    return validate_reference_trace_spec(load_reference_trace_spec(name))


def validate_reference_trace_spec(spec: ReferenceTraceSpec) -> TraceValidationReport:
    """Validate one in-memory reference-trace specification.

    Parameters
    ----------
    spec:
        Reference-trace specification to simulate and compare.

    Returns
    -------
    TraceValidationReport
        Feature-level validation report.
    """
    simulation = simulate_reference_trace(spec)
    mismatches: list[FeatureMismatch] = []
    for feature, expected in spec.expected_features.items():
        actual = simulation.features[feature]
        tolerance = spec.tolerances[feature]
        if not tolerance.accepts(actual, expected):
            mismatches.append(
                FeatureMismatch(
                    feature=feature,
                    expected=expected,
                    actual=actual,
                    tolerance=tolerance,
                    absolute_error=abs(actual - expected),
                )
            )
    return TraceValidationReport(
        name=spec.name,
        passed=not mismatches,
        simulation=simulation,
        mismatches=tuple(mismatches),
    )


def validate_all_reference_traces() -> tuple[TraceValidationReport, ...]:
    """Validate every committed reference trace.

    Returns
    -------
    tuple[TraceValidationReport, ...]
        Sorted reports for the current corpus.
    """
    return tuple(validate_reference_trace(name) for name in list_reference_trace_specs())


def _coerce_spec(spec_or_name: ReferenceTraceSpec | str) -> ReferenceTraceSpec:
    if isinstance(spec_or_name, ReferenceTraceSpec):
        return spec_or_name
    return load_reference_trace_spec(spec_or_name)


def _extract_features(
    trace: Mapping[str, tuple[float, ...]], spikes: tuple[int, ...]
) -> Mapping[str, float]:
    features: dict[str, float] = {
        "spike_count": float(math.fsum(spikes)),
        "first_spike_step": float(
            next((index for index, spike in enumerate(spikes, start=1) if spike), -1)
        ),
    }
    for variable, values in trace.items():
        features[f"final.{variable}"] = values[-1]
        features[f"min.{variable}"] = min(values)
        features[f"max.{variable}"] = max(values)
        features[f"mean.{variable}"] = math.fsum(values) / len(values)
    return MappingProxyType(features)


__all__ = [
    "simulate_reference_trace",
    "validate_all_reference_traces",
    "validate_reference_trace",
    "validate_reference_trace_spec",
]
