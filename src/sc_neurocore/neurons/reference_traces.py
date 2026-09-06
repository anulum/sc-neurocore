# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuron reference-trace validation harness

"""Public facade for schema-driven neuron reference-trace validation.

Three surfaces meet here: the corpus itself, the independence adjudication
that says what each trace is a reference *for*, and the negative controls
that demonstrate a trace can fail when the model is wrong.
"""

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
from sc_neurocore.neurons.reference_trace_mutations import (
    MUTATIONS,
    REFERENCE_TRACE_CONTROL_VERSION,
    ControlOutcome,
    NegativeControlReport,
    negative_control_report,
    run_negative_control,
)
from sc_neurocore.neurons.reference_trace_provenance import (
    REFERENCE_TRACE_ADJUDICATION_VERSION,
    CorpusAdjudication,
    ReferenceTraceAdjudicationError,
    SpecAdjudication,
    adjudicate_corpus,
    adjudicate_spec,
)
from sc_neurocore.neurons.reference_trace_io import (
    list_reference_trace_specs,
    load_reference_trace_spec,
    reference_trace_spec_from_payload,
)
from sc_neurocore.neurons.reference_trace_runner import (
    extract_trace_features,
    simulate_reference_trace,
    validate_all_reference_traces,
    validate_reference_trace,
    validate_reference_trace_spec,
)


__all__ = [
    "ControlOutcome",
    "CorpusAdjudication",
    "FeatureMismatch",
    "FeatureTolerance",
    "MUTATIONS",
    "NegativeControlReport",
    "REFERENCE_TRACE_ADJUDICATION_VERSION",
    "REFERENCE_TRACE_CONTROL_VERSION",
    "REFERENCE_TRACE_SCHEMA_VERSION",
    "ReferenceTraceAdjudicationError",
    "ReferenceTraceProtocol",
    "ReferenceTraceProvenance",
    "ReferenceTraceSpec",
    "SpecAdjudication",
    "TraceSimulationResult",
    "TraceValidationReport",
    "adjudicate_corpus",
    "adjudicate_spec",
    "extract_trace_features",
    "list_reference_trace_specs",
    "load_reference_trace_spec",
    "negative_control_report",
    "reference_trace_spec_from_payload",
    "run_negative_control",
    "simulate_reference_trace",
    "validate_all_reference_traces",
    "validate_reference_trace",
    "validate_reference_trace_spec",
]
