# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuron reference-trace validation harness

"""Public facade for schema-driven neuron reference-trace validation."""

from __future__ import annotations

from sc_neurocore.neurons.reference_trace_contracts import (
    REFERENCE_TRACE_SCHEMA_VERSION,
    FeatureMismatch,
    FeatureTolerance,
    ReferenceTraceProtocol,
    ReferenceTraceProvenance,
    ReferenceTraceSpec,
    TraceSimulationResult,
    TraceValidationReport,
)
from sc_neurocore.neurons.reference_trace_io import (
    list_reference_trace_specs,
    load_reference_trace_spec,
    reference_trace_spec_from_payload,
)
from sc_neurocore.neurons.reference_trace_runner import (
    simulate_reference_trace,
    validate_all_reference_traces,
    validate_reference_trace,
    validate_reference_trace_spec,
)


__all__ = [
    "FeatureMismatch",
    "FeatureTolerance",
    "REFERENCE_TRACE_SCHEMA_VERSION",
    "ReferenceTraceProtocol",
    "ReferenceTraceProvenance",
    "ReferenceTraceSpec",
    "TraceSimulationResult",
    "TraceValidationReport",
    "list_reference_trace_specs",
    "load_reference_trace_spec",
    "reference_trace_spec_from_payload",
    "simulate_reference_trace",
    "validate_all_reference_traces",
    "validate_reference_trace",
    "validate_reference_trace_spec",
]
