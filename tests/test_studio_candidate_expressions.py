# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate expression call contracts

"""Exercise candidate call admission and accepted numerical overloads."""

from __future__ import annotations

import pytest

from sc_neurocore.studio.candidate_diff import diff_candidate
from sc_neurocore.studio.candidate_package import validate_candidate
from sc_neurocore.studio.candidate_run import simulate_candidate
from tests.studio_candidate_support import adex_candidate


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("sqrt()", "function 'sqrt' does not accept 0 positional arguments"),
        ("sqrt(v, 0)", "function 'sqrt' does not accept 2 positional arguments"),
        ("exp()", "function 'exp' does not accept 0 positional arguments"),
        ("exp(v, None, None)", "function 'exp' does not accept 3 positional arguments"),
        ("abs()", "function 'abs' does not accept 0 positional arguments"),
        ("exprel()", "function 'exprel' does not accept 0 positional arguments"),
        ("sigmoid()", "function 'sigmoid' does not accept 0 positional arguments"),
        ("clip(v)", "function 'clip' does not accept 1 positional arguments"),
        ("clip(v, 0)", "function 'clip' does not accept 2 positional arguments"),
        ("min()", "function 'min' does not accept 0 positional arguments"),
        ("max()", "function 'max' does not accept 0 positional arguments"),
        ("v()", "'v' is not an equation function"),
        ("pi()", "'pi' is not an equation function"),
    ],
)
def test_bad_calls_are_refused_at_the_actual_equation_field(expression: str, reason: str) -> None:
    """Reject an unevaluable call before accepting the candidate model."""
    document = adex_candidate()
    document["model"]["dynamics"]["v"] = expression
    validation = validate_candidate(document)
    assert not validation.valid
    assert len(validation.diagnostics) == 1
    assert validation.diagnostics[0].location == "/model/dynamics/v"
    assert validation.diagnostics[0].message == reason


@pytest.mark.parametrize(
    "expression",
    [
        "sqrt(4)",
        "abs(v)",
        "exprel(0)",
        "sigmoid(0)",
        "clip(v, -70, -50)",
        "clip(v, None, 0)",
        "min([v, 0])",
        "max(v, 0)",
        *[f"{name}(0)" for name in ("exp", "log", "sin", "cos", "tanh", "cosh", "sinh")],
        *[f"{name}(1, None)" for name in ("exp", "log", "sin", "cos", "tanh", "cosh", "sinh")],
    ],
)
def test_supported_overloads_are_admitted_and_reach_the_real_runner(expression: str) -> None:
    """Retain actual NumPy outputs, iterable extrema and helper signatures."""
    document = adex_candidate()
    document["model"]["dynamics"]["v"] = expression
    assert validate_candidate(document).valid
    result = simulate_candidate(document, current=0.0, steps=1)
    if expression == "log(0)":
        assert result["diverged_at_step"] == 0
    else:
        assert result["diverged_at_step"] is None
    assert diff_candidate(document)["status"] == "compared"


def test_nested_and_attribute_calls_keep_existing_scalar_semantics() -> None:
    """Keep nested helper calls and the float attribute API executable."""
    document = adex_candidate()
    document["model"]["dynamics"]["v"] = "abs(min(v.real, -60))"
    assert validate_candidate(document).valid
    assert simulate_candidate(document, current=0.0, steps=1)["diverged_at_step"] is None
    assert diff_candidate(document)["dynamics"][0]["status"] == "not_comparable"
