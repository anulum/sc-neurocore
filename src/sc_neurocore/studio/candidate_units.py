# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Located dimensional admission of candidate model packages

"""Bind candidate unit declarations to every expression the DSL executes.

Admission runs after structural, symbol, arity and numerical-profile checks.
It does not instantiate a strict-units runtime or normalise magnitudes: the
candidate's numbers, profile, digest and random stream remain untouched.
This establishes dimensional and numeric-scale consistency, not scientific
agreement with a publication, stability, hardware fidelity or promotion.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.neurons.model_profile import resolve_profile
from sc_neurocore.studio.candidate_unit_expressions import UnitExpression
from sc_neurocore.studio.candidate_unit_values import (
    CandidateUnitRefused,
    Unit,
    coherent,
    dimensionless,
    parse_unit,
    product,
)


def _pointer(*parts: str) -> str:
    """Escape a candidate field location using JSON Pointer segment rules."""
    return "/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in parts)


def candidate_unit_diagnostics(document: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Check units in an otherwise admitted candidate and locate every failure.

    Parameters
    ----------
    document : mapping
        Candidate JSON already accepted by the structural and DSL gates. State
        and parameter numbers retain their declared magnitudes; units supply
        metadata, never a conversion request.

    Returns
    -------
    tuple of pairs
        JSON pointers and deliberately authored refusal messages. Dynamics are
        derivatives for ODE profiles and next-state values for maps; resets
        retain state units, rates use inverse declared time, probabilities and
        predicates are unscaled dimensionless. Graded outputs may be physical
        scalars, with all internal arithmetic checked. Declared profile units
        must agree with the package in dimension and scale.
    """
    problems: list[tuple[str, str]] = []
    symbols: dict[str, Unit] = {}
    units = document["units"]
    model = document["model"]
    for section in ("state", "parameters"):
        for name, text in units[section].items():
            try:
                unit = parse_unit(text)
                constant = (
                    Fraction(str(model[section][name]))
                    if section == "parameters" and not unit.dimensions and unit.scale == 1.0
                    else None
                )
                symbols[name] = Unit(unit.dimensions, unit.scale, constant)
            except CandidateUnitRefused as exc:
                problems.append((_pointer("units", section, name), str(exc)))
    input_units: dict[str, Unit] = {}
    for field in ("current", "time"):
        try:
            input_units[field] = parse_unit(units[field])
        except CandidateUnitRefused as exc:
            problems.append((_pointer("units", field), str(exc)))
    if problems:
        return tuple(problems)
    symbols["I"] = input_units["current"]
    time = input_units["time"]
    numerical = resolve_profile(model).numerical
    if numerical.family != "map":
        try:
            # Only the dimension is constrained here: ms, s and other finite
            # physical time scales remain valid when the equations agree.
            coherent(Unit(time.dimensions), Unit(parse_unit("s").dimensions))
        except CandidateUnitRefused:
            problems.append(("/units/time", "ODE integration time must have the dimension of time"))
    raw_profile = model.get("profile")
    profile = raw_profile if isinstance(raw_profile, Mapping) else {}
    raw_profile_units = profile.get("units")
    profile_units = raw_profile_units if isinstance(raw_profile_units, Mapping) else {}
    for name, text in profile_units.items():
        try:
            coherent(symbols[name], parse_unit(text))
        except CandidateUnitRefused as exc:
            section = "state" if name in model["state"] else "parameters"
            problems.append(
                (
                    _pointer("units", section, name),
                    "the model profile and candidate unit disagree: " + str(exc),
                )
            )
    if numerical.time_unit == "iteration":
        try:
            dimensionless(time)
        except CandidateUnitRefused:
            problems.append(
                (
                    "/units/time",
                    "an iteration-only map requires an unscaled dimensionless time unit",
                )
            )
    elif numerical.time_unit:
        try:
            coherent(time, parse_unit(numerical.time_unit))
        except CandidateUnitRefused as exc:
            problems.append(
                (
                    "/units/time",
                    "the numerical profile and candidate time unit disagree: " + str(exc),
                )
            )
    for name, value in build_eval_namespace().items():
        if not callable(value) and name not in symbols:
            symbols[name] = Unit(constant=Fraction(str(value)))
    symbols["xi"] = time.power(Fraction(-1, 2))
    expressions = UnitExpression(symbols)

    def check(expression: str, location: str, expected: Unit | None) -> None:
        """Collect a deliberately authored expression refusal at its field."""
        try:
            inferred = expressions.infer(expression)
            if expected is not None:
                coherent(inferred, expected)
        except CandidateUnitRefused as exc:
            problems.append((location, str(exc)))

    for name, expression in model["dynamics"].items():
        location = _pointer("model", "dynamics", name)
        try:
            expected = symbols[name]
            if numerical.family != "map":
                expected = product(expected, time, divide=True)
        except CandidateUnitRefused as exc:
            problems.append((location, str(exc)))
        else:
            check(expression, location, expected)
    for name, expression in (model.get("reset") or {}).items():
        check(expression, _pointer("model", "reset", name), symbols[name])
    threshold = model.get("threshold", {})
    for field, event_unit in (
        ("condition", Unit()),
        ("rate_expression", None),
        ("probability_expression", Unit()),
        ("spike_value", None),
    ):
        expression = threshold.get(field)
        if expression is not None and not (field == "condition" and expression == "stochastic"):
            location = _pointer("model", "threshold", field)
            try:
                if field == "rate_expression":
                    event_unit = product(Unit(), time, divide=True)
            except CandidateUnitRefused as exc:
                problems.append((location, str(exc)))
            else:
                check(expression, location, event_unit)
    return tuple(problems)
