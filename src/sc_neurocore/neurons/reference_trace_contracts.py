# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuron reference-trace validation contracts

"""Immutable contracts for neuron reference-trace validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType

REFERENCE_TRACE_SCHEMA_VERSION = "sc-neurocore.reference-trace.v1"


@dataclass(frozen=True, slots=True)
class FeatureTolerance:
    """Absolute and relative tolerance for one scalar trace feature.

    Parameters
    ----------
    absolute:
        Absolute error allowed between simulated and reference feature values.
    relative:
        Relative error allowed as a fraction of the reference value magnitude.
    """

    absolute: float
    relative: float = 0.0

    def accepts(self, actual: float, expected: float) -> bool:
        """Return whether ``actual`` lies inside the configured tolerance.

        Parameters
        ----------
        actual:
            Feature value extracted from the current simulation.
        expected:
            Feature value recorded by the reference corpus.

        Returns
        -------
        bool
            ``True`` when the absolute error is no larger than
            ``max(absolute, relative * abs(expected))``.
        """
        limit = max(self.absolute, self.relative * abs(expected))
        return math.isfinite(actual) and abs(actual - expected) <= limit


@dataclass(frozen=True, slots=True)
class ReferenceTraceProvenance:
    """Provenance attached to one reference trace.

    Parameters
    ----------
    kind:
        Reference class, for example ``analytic_closed_form``.
    source:
        Human-readable source for the reference values.
    equation:
        Equation or feature-generation statement used to derive the values.
    citation:
        Optional publication or DOI string when the trace is external.
    """

    kind: str
    source: str
    equation: str
    citation: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceTraceProtocol:
    """Simulation protocol for one reference trace.

    Parameters
    ----------
    dt:
        Simulation timestep in the units used by the model schema.
    steps:
        Number of timesteps to execute.
    inputs:
        Constant keyword inputs passed to ``UniversalNeuron.step``.
    state_variables:
        State variables to record after each timestep. May be empty for a
        genuinely stateless deterministic threshold model; event features are
        still recorded.
    parameter_overrides:
        Optional schema parameter values that define the enrolled operating
        profile without changing the bundled schema defaults.
    """

    dt: float
    steps: int
    inputs: Mapping[str, float]
    state_variables: tuple[str, ...]
    parameter_overrides: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class ReferenceTraceSpec:
    """Committed reference-trace specification.

    Parameters
    ----------
    name:
        Stable corpus identifier.
    schema_name:
        Bundled UniversalNeuron schema name.
    runner:
        Production runner name. The v1 corpus supports ``universal_dsl``.
    protocol:
        Deterministic simulation protocol.
    provenance:
        Reference source and derivation metadata.
    expected_features:
        Scalar reference features keyed by stable feature names.
    tolerances:
        Per-feature tolerance map.
    """

    name: str
    schema_name: str
    runner: str
    protocol: ReferenceTraceProtocol
    provenance: ReferenceTraceProvenance
    expected_features: Mapping[str, float]
    tolerances: Mapping[str, FeatureTolerance]


@dataclass(frozen=True, slots=True)
class TraceSimulationResult:
    """Trace and feature values produced by the current implementation.

    Parameters
    ----------
    name:
        Reference-trace name that was simulated.
    steps:
        Number of executed timesteps.
    trace:
        Recorded state values after each timestep.
    spikes:
        Spike indicator emitted by each timestep.
    features:
        Scalar feature map extracted from ``trace`` and ``spikes``.
    """

    name: str
    steps: int
    trace: Mapping[str, tuple[float, ...]]
    spikes: tuple[int, ...]
    features: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class FeatureMismatch:
    """One failed scalar-feature comparison.

    Parameters
    ----------
    feature:
        Feature name whose value drifted.
    expected:
        Reference feature value.
    actual:
        Current simulation feature value.
    tolerance:
        Tolerance that was applied.
    absolute_error:
        Absolute difference between ``actual`` and ``expected``.
    """

    feature: str
    expected: float
    actual: float
    tolerance: FeatureTolerance
    absolute_error: float


@dataclass(frozen=True, slots=True)
class TraceValidationReport:
    """Validation report for one reference trace.

    Parameters
    ----------
    name:
        Reference-trace name.
    passed:
        Whether every expected feature matched within tolerance.
    simulation:
        Trace produced by the current implementation.
    mismatches:
        Failed feature comparisons, empty on success.
    """

    name: str
    passed: bool
    simulation: TraceSimulationResult
    mismatches: tuple[FeatureMismatch, ...]


__all__ = [
    "FeatureMismatch",
    "FeatureTolerance",
    "REFERENCE_TRACE_SCHEMA_VERSION",
    "ReferenceTraceProtocol",
    "ReferenceTraceProvenance",
    "ReferenceTraceSpec",
    "TraceSimulationResult",
    "TraceValidationReport",
]
