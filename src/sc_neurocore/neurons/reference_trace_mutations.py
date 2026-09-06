# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference-trace negative controls

"""Show that a reference trace fails when the model is wrong.

A corpus that passes proves nothing on its own: it has to be shown to fail when
the thing it validates is broken. These are the controls, one per error class a
neuron model actually suffers — a flipped drive sign, a unit scale slipped by
three orders of magnitude, a drive dropped entirely, events recorded on the
wrong side of the state update, and a moved threshold surface.

Each control is applied to a *copy* of the specification or to a deliberately
mutated runner; nothing here can change the committed corpus. Four outcomes are
possible and all four are reported:

``detected``
    The mutated run violated the trace's own tolerances. The control bit.
``refused``
    The mutated run could not complete — the model rejected the mutated input or
    left its numeric domain. A wrong answer was still caught, by refusal.
``inapplicable``
    The mutation provably changes nothing about *this* protocol, decided before
    running it: negating a zero drive, or shifting a threshold a trace never
    reaches. Reported as a boundary, never counted as a pass.
``undetected``
    The mutation was applied and the trace still passed. Recorded plainly. A
    scale-invariant logical unit is the honest example: multiplying its input
    above threshold cannot change its truth table.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from contextlib import contextmanager
import json
import math
import re
from types import MappingProxyType
from typing import Literal
import warnings

from collections.abc import Iterator

import numpy as np

from sc_neurocore.neurons.reference_trace_contracts import (
    ReferenceTraceProtocol,
    ReferenceTraceSpec,
)
from sc_neurocore.neurons.reference_trace_runner import (
    extract_trace_features,
    validate_reference_trace_spec,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

#: Negative-control contract version.
REFERENCE_TRACE_CONTROL_VERSION = "sc-neurocore.reference-trace-controls.v1"

MutationName = Literal[
    "sign",
    "unit_scale",
    "omitted_current",
    "event_ordering",
    "threshold_shift",
]

MutationStatus = Literal["detected", "refused", "undetected", "inapplicable"]

#: Every control, in the order a report lists them.
MUTATIONS: tuple[MutationName, ...] = (
    "sign",
    "unit_scale",
    "omitted_current",
    "event_ordering",
    "threshold_shift",
)

#: Factor a mistaken millivolt/volt or nanoamp/microamp conversion introduces.
_UNIT_SCALE_FACTOR = 1000.0

#: Relative threshold displacements tried in order, largest last. A trace that
#: resolves none of these does not pin its threshold surface at all.
_THRESHOLD_FRACTIONS: tuple[float, ...] = (0.1, 0.3, 1.0)


@contextmanager
def _broken_model_is_expected() -> Iterator[None]:
    """Silence the numeric complaints a deliberately broken run produces.

    A mutant is meant to be wrong, and a wrong drive routinely leaves the
    model's numeric domain — overflow in an exponential, a divide by a value
    that should never have been zero. The refusal is the finding; the warning
    is noise, and letting it escape would train a reader to ignore warnings
    from the production validator too.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            yield


@dataclass(frozen=True, slots=True)
class ControlOutcome:
    """Result of applying one negative control to one reference trace.

    Attributes
    ----------
    name : str
        Corpus identifier of the trace under control.
    mutation : str
        Which control was applied.
    status : str
        ``detected``, ``refused``, ``undetected`` or ``inapplicable``.
    reason : str
        Why the control reached that status, in words an operator can act on.
    mismatched_features : int
        How many features violated tolerance; zero unless ``detected``.
    """

    name: str
    mutation: MutationName
    status: MutationStatus
    reason: str
    mismatched_features: int = 0

    @property
    def caught(self) -> bool:
        """Return whether this control demonstrated the trace catches the error."""
        return self.status in {"detected", "refused"}

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-safe row for reports and documentation."""
        return {
            "mismatched_features": self.mismatched_features,
            "mutation": self.mutation,
            "name": self.name,
            "reason": self.reason,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class NegativeControlReport:
    """Every control outcome for the committed corpus.

    Attributes
    ----------
    version : str
        Negative-control contract version.
    outcomes : tuple of ControlOutcome
        One outcome per trace and control.
    """

    version: str
    outcomes: tuple[ControlOutcome, ...]

    def for_trace(self, name: str) -> tuple[ControlOutcome, ...]:
        """Return every outcome recorded for one trace."""
        return tuple(outcome for outcome in self.outcomes if outcome.name == name)

    def with_status(self, status: MutationStatus) -> tuple[ControlOutcome, ...]:
        """Return every outcome with one status."""
        return tuple(outcome for outcome in self.outcomes if outcome.status == status)

    def uncontrolled_traces(self) -> tuple[str, ...]:
        """Return traces no control could catch — a vacuous trace, if any."""
        names = sorted({outcome.name for outcome in self.outcomes})
        return tuple(name for name in names if not any(o.caught for o in self.for_trace(name)))

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-safe report of every control outcome."""
        return {
            "outcomes": [outcome.to_public_dict() for outcome in self.outcomes],
            "uncontrolled_traces": list(self.uncontrolled_traces()),
            "version": self.version,
        }


def run_negative_control(spec: ReferenceTraceSpec, mutation: MutationName) -> ControlOutcome:
    """Apply one negative control to one reference trace.

    Parameters
    ----------
    spec : ReferenceTraceSpec
        The trace to control. It is never modified; mutations act on copies.
    mutation : str
        One of :data:`MUTATIONS`.

    Returns
    -------
    ControlOutcome
        What the control demonstrated, including why it could not apply.

    Raises
    ------
    ValueError
        ``mutation`` is not a known control.
    """
    if mutation not in MUTATIONS:
        raise ValueError(f"unknown negative control {mutation!r}")
    if mutation == "event_ordering":
        return _event_ordering_control(spec)
    if mutation == "threshold_shift":
        return _threshold_shift_control(spec)
    return _drive_control(spec, mutation)


def negative_control_report(names: Iterable[str] | None = None) -> NegativeControlReport:
    """Apply every control to every deterministic trace in the corpus.

    Parameters
    ----------
    names : iterable of str, optional
        Restrict the report to these traces. Defaults to the whole corpus.

    Returns
    -------
    NegativeControlReport
        One outcome per trace and control, in corpus order.
    """
    from sc_neurocore.neurons.reference_trace_io import (
        list_reference_trace_specs,
        load_reference_trace_spec,
    )

    selected = sorted(names) if names is not None else sorted(list_reference_trace_specs())
    outcomes: list[ControlOutcome] = []
    for name in selected:
        spec = load_reference_trace_spec(name)
        outcomes.extend(run_negative_control(spec, mutation) for mutation in MUTATIONS)
    return NegativeControlReport(version=REFERENCE_TRACE_CONTROL_VERSION, outcomes=tuple(outcomes))


def _drive_control(spec: ReferenceTraceSpec, mutation: MutationName) -> ControlOutcome:
    """Flip, rescale or remove the drive and see whether the trace notices."""
    inputs = dict(spec.protocol.inputs)
    if not inputs or all(value == 0.0 for value in inputs.values()):
        return ControlOutcome(
            name=spec.name,
            mutation=mutation,
            status="inapplicable",
            reason="the protocol drives the model with zero input, so this mutation is a no-op",
        )
    if mutation == "sign":
        mutated = {key: -value for key, value in inputs.items()}
    elif mutation == "unit_scale":
        mutated = {key: value * _UNIT_SCALE_FACTOR for key, value in inputs.items()}
    else:
        mutated = dict.fromkeys(inputs, 0.0)
    return _verdict(spec, _with_inputs(spec, mutated), mutation, _drive_reason(mutation))


def _threshold_shift_control(spec: ReferenceTraceSpec) -> ControlOutcome:
    """Move the threshold surface and see whether the trace's events move."""
    if spec.expected_features.get("spike_count", 0.0) == 0.0:
        return ControlOutcome(
            name=spec.name,
            mutation="threshold_shift",
            status="inapplicable",
            reason="the reference records no event, so this protocol never reaches the threshold",
        )
    parameter = _threshold_parameter(spec)
    if parameter is None:
        return ControlOutcome(
            name=spec.name,
            mutation="threshold_shift",
            status="inapplicable",
            reason="the schema threshold condition names no single adjustable parameter",
        )
    key, base = parameter
    for fraction in _THRESHOLD_FRACTIONS:
        shifted = base * (1.0 + fraction) if base else fraction
        outcome = _verdict(
            spec,
            _with_parameter(spec, key, shifted),
            "threshold_shift",
            f"{key} displaced by {fraction:.0%} from {base:g} to {shifted:g}",
        )
        if outcome.caught:
            return outcome
    return ControlOutcome(
        name=spec.name,
        mutation="threshold_shift",
        status="undetected",
        reason=(
            f"{key} survives displacement to {_THRESHOLD_FRACTIONS[-1]:.0%}; this trace does not "
            "resolve its threshold surface"
        ),
    )


def _event_ordering_control(spec: ReferenceTraceSpec) -> ControlOutcome:
    """Record state before the update instead of after, and compare."""
    neuron = UniversalNeuron.from_schema(
        spec.schema_name,
        dt_override=spec.protocol.dt,
        parameter_overrides=dict(spec.protocol.parameter_overrides),
    )
    recorded: dict[str, list[float]] = {name: [] for name in spec.protocol.state_variables}
    spikes: list[int] = []
    try:
        with _broken_model_is_expected():
            for _ in range(spec.protocol.steps):
                for variable in spec.protocol.state_variables:
                    recorded[variable].append(float(neuron.state[variable]))
                spikes.append(neuron.step(**dict(spec.protocol.inputs)))
    except Exception as exc:  # noqa: BLE001 - any refusal is a caught error
        return ControlOutcome(
            name=spec.name,
            mutation="event_ordering",
            status="refused",
            reason=f"the mutated ordering could not complete ({type(exc).__name__})",
        )
    features = extract_trace_features(
        {variable: tuple(values) for variable, values in recorded.items()}, tuple(spikes)
    )
    mismatches = sum(
        1
        for feature, expected in spec.expected_features.items()
        if not spec.tolerances[feature].accepts(features.get(feature, math.nan), expected)
    )
    if mismatches:
        return ControlOutcome(
            name=spec.name,
            mutation="event_ordering",
            status="detected",
            reason="recording state before the update instead of after violates the trace",
            mismatched_features=mismatches,
        )
    return ControlOutcome(
        name=spec.name,
        mutation="event_ordering",
        status="undetected",
        reason="the recorded features do not distinguish pre-update from post-update state",
    )


def _verdict(
    spec: ReferenceTraceSpec,
    mutant: ReferenceTraceSpec,
    mutation: MutationName,
    reason: str,
) -> ControlOutcome:
    try:
        with _broken_model_is_expected():
            report = validate_reference_trace_spec(mutant)
    except Exception as exc:  # noqa: BLE001 - any refusal is a caught error
        return ControlOutcome(
            name=spec.name,
            mutation=mutation,
            status="refused",
            reason=f"{reason}; the mutated run could not complete ({type(exc).__name__})",
        )
    if report.passed:
        return ControlOutcome(
            name=spec.name,
            mutation=mutation,
            status="undetected",
            reason=f"{reason}; the trace still passed",
        )
    return ControlOutcome(
        name=spec.name,
        mutation=mutation,
        status="detected",
        reason=reason,
        mismatched_features=len(report.mismatches),
    )


def _drive_reason(mutation: MutationName) -> str:
    if mutation == "sign":
        return "the drive sign was flipped"
    if mutation == "unit_scale":
        return f"the drive was rescaled by {_UNIT_SCALE_FACTOR:g}"
    return "the drive was removed"


def _with_inputs(spec: ReferenceTraceSpec, inputs: Mapping[str, float]) -> ReferenceTraceSpec:
    protocol = spec.protocol
    return replace(
        spec,
        protocol=ReferenceTraceProtocol(
            dt=protocol.dt,
            steps=protocol.steps,
            inputs=MappingProxyType(dict(inputs)),
            state_variables=protocol.state_variables,
            parameter_overrides=protocol.parameter_overrides,
        ),
    )


def _with_parameter(spec: ReferenceTraceSpec, key: str, value: float) -> ReferenceTraceSpec:
    protocol = spec.protocol
    overrides = dict(protocol.parameter_overrides)
    overrides[key] = value
    return replace(
        spec,
        protocol=ReferenceTraceProtocol(
            dt=protocol.dt,
            steps=protocol.steps,
            inputs=protocol.inputs,
            state_variables=protocol.state_variables,
            parameter_overrides=MappingProxyType(overrides),
        ),
    )


def _threshold_parameter(spec: ReferenceTraceSpec) -> tuple[str, float] | None:
    """Return the single parameter the schema's threshold condition names."""
    neuron = UniversalNeuron.from_schema(
        spec.schema_name,
        dt_override=spec.protocol.dt,
        parameter_overrides=dict(spec.protocol.parameter_overrides),
    )
    document = json.loads(neuron.to_json())
    threshold = document.get("threshold")
    if not isinstance(threshold, Mapping):
        return None
    condition = threshold.get("condition")
    parameters = document.get("parameters")
    if not isinstance(condition, str) or not isinstance(parameters, Mapping):
        return None
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", condition))
    named = sorted(identifiers & set(parameters))
    if len(named) != 1:
        return None
    key = named[0]
    overrides = dict(spec.protocol.parameter_overrides)
    if key in overrides:
        return key, float(overrides[key])
    declared = parameters[key]
    if isinstance(declared, bool) or not isinstance(declared, (int, float)):
        # A threshold expressed as anything but a number cannot be displaced by
        # a fraction of itself; the control reports that instead of guessing.
        return None
    return key, float(declared)


__all__ = [
    "MUTATIONS",
    "REFERENCE_TRACE_CONTROL_VERSION",
    "ControlOutcome",
    "MutationName",
    "MutationStatus",
    "NegativeControlReport",
    "negative_control_report",
    "run_negative_control",
]
