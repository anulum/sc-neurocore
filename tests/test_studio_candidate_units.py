# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public dimensional candidate admission and magnitude custody

"""Check real candidates without changing numerical model values or profiles."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any

import pytest

from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from sc_neurocore.studio.candidate_package import candidate_sha256, validate_candidate
from sc_neurocore.studio.candidate_run import CandidateRejected, review_packet, simulate_candidate
from tests.studio_candidate_support import adex_candidate


def test_adex_admission_preserves_every_number_and_the_actual_dsl_trace() -> None:
    """Compare an admitted candidate with the unchanged numeric DSL, bit for bit."""
    document = adex_candidate()
    original = copy.deepcopy(document)
    digest = candidate_sha256(document)
    direct = UniversalNeuron.from_dict(document["model"])
    expected: dict[str, list[float]] = {name: [] for name in direct.state}
    events: list[int] = []
    for step in range(300):
        if direct.step(I=800.0):
            events.append(step)
        for name, value in direct.state.items():
            expected[name].append(float(value))
    validation = validate_candidate(document)
    assert validation.valid, validation.diagnostics
    result = simulate_candidate(document, current=800.0, steps=300)
    assert result["trace"] == expected
    assert result["spike_steps"] == events
    assert result["profile"] == direct.realised_profile()
    packet = review_packet(document)
    assert packet["candidate"] == original == document
    assert packet["candidate_sha256"] == digest == candidate_sha256(document)
    assert all("dimensional consistency" not in text for text in packet["not_established"])
    assert any("independent validation" in text for text in packet["not_established"])


@pytest.mark.parametrize("profile", [True, False])
def test_voltage_relabelled_as_time_is_refused_before_any_run(profile: bool) -> None:
    """Reject wrong dimensions with and without duplicate profile declarations.

    Parameters
    ----------
    profile : bool
        Keep the bundled profile declaration or exercise equations alone.
    """
    document = adex_candidate()
    if not profile:
        document["model"].pop("profile")
    document["units"]["state"]["v"] = "ms"
    validation = validate_candidate(document)
    assert not validation.valid
    assert any(row.location == "/model/dynamics/v" for row in validation.diagnostics)
    assert any(row.location == "/model/threshold/condition" for row in validation.diagnostics)
    actions: tuple[Callable[[], object], ...] = (
        lambda: simulate_candidate(document, current=800.0, steps=1),
        lambda: review_packet(document),
    )
    for action in actions:
        with pytest.raises(CandidateRejected) as refused:
            action()
        assert refused.value.validation == validation.to_public_dict()


@pytest.mark.parametrize(
    ("section", "name", "value", "location", "fragment"),
    [
        ("dynamics", "v", "(v + I) / tau", "/model/dynamics/v", "incompatible dimensions"),
        ("dynamics", "v", "exp(v) * delta_T / tau", "/model/dynamics/v", "incompatible dimensions"),
        (
            "dynamics",
            "v",
            "v / tau if True else I / tau",
            "/model/dynamics/v",
            "incompatible dimensions",
        ),
        (
            "dynamics",
            "v",
            "v / tau if (True or I) else v / tau",
            "/model/dynamics/v",
            "incompatible dimensions",
        ),
        ("reset", "w", "v", "/model/reset/w", "incompatible dimensions"),
        (
            "threshold",
            "condition",
            "v >= I",
            "/model/threshold/condition",
            "incompatible dimensions",
        ),
        ("threshold", "condition", "v", "/model/threshold/condition", "incompatible dimensions"),
        (
            "threshold",
            "spike_value",
            "v + I",
            "/model/threshold/spike_value",
            "incompatible dimensions",
        ),
    ],
)
def test_all_authored_expression_fields_enforce_units(
    section: str, name: str, value: str, location: str, fragment: str
) -> None:
    """Locate a mismatch even in a branch the numeric runner would not visit.

    Parameters
    ----------
    section, name : str
        Actual executable model expression field.
    value : str
        Safe expression with an incompatible dimensional path.
    location, fragment : str
        Expected JSON pointer and authored diagnostic.
    """
    document = adex_candidate()
    document["model"][section][name] = value
    validation = validate_candidate(document)
    assert not validation.valid
    assert any(
        row.location == location and fragment in row.message for row in validation.diagnostics
    )


@pytest.mark.parametrize("field", ["rate_expression", "probability_expression"])
@pytest.mark.parametrize("valid", [True, False])
def test_stochastic_threshold_expressions_are_checked(field: str, valid: bool) -> None:
    """Check rate and probability dimensions through real candidate admission.

    Parameters
    ----------
    field : str
        Escape-rate or Poisson probability expression.
    valid : bool
        Use the required inverse-time or dimensionless unit, or physical voltage.
    """
    document = adex_candidate()
    rate = field == "rate_expression"
    document["model"]["threshold"] = {
        "condition": "stochastic",
        "detection": "escape_rate" if rate else "poisson",
        field: ("1 / tau" if rate else "0.1") if valid else "v",
    }
    validation = validate_candidate(document)
    assert validation.valid is valid, validation.diagnostics
    if valid:
        assert simulate_candidate(document, current=0.0, steps=3)["diverged_at_step"] is None
    else:
        assert any(row.location == f"/model/threshold/{field}" for row in validation.diagnostics)


def test_map_equations_are_next_state_values_and_can_reset_to_literal_zero() -> None:
    """Avoid applying derivative units to a genuine declared map profile."""
    document = adex_candidate()
    model = document["model"]
    model.pop("profile")
    model["integration"]["method"] = "map"
    model.pop("extensions")
    document["units"]["time"] = "dimensionless"
    model["dynamics"] = {"v": "v_rest", "w": "w + b_adapt"}
    model["reset"] = {"v": "0", "w": "0"}
    assert validate_candidate(document).valid
    result = simulate_candidate(document, current=0.0, steps=1)
    assert result["final_state"] == {"v": -65.0, "w": 7.0}
    document["units"]["time"] = "ms"
    refused = validate_candidate(document)
    assert any(
        row.location == "/units/time" and "iteration-only map" in row.message
        for row in refused.diagnostics
    )
    model["profile"] = {"contract": "sc-neurocore.model-profile.v1", "time_unit": "ms"}
    assert validate_candidate(document).valid
    model["dynamics"]["v"] = "v / tau"
    assert not validate_candidate(document).valid


@pytest.mark.parametrize(
    ("path", "value", "location", "fragment"),
    [
        (("units", "state", "v"), "V", "/units/state/v", "unit scales"),
        (("units", "time"), "s", "/units/time", "time unit disagree"),
        (("units", "time"), "mV", "/units/time", "dimension of time"),
        (("units", "current"), "degC", "/units/current", "multiplicative unit"),
        (("units", "state", "w"), "dB", "/units/state/w", "multiplicative unit"),
        (
            ("model", "profile", "units", "v"),
            "unknown_quantity",
            "/units/state/v",
            "profile and candidate unit disagree",
        ),
    ],
)
def test_unit_metadata_cannot_request_implicit_conversion(
    path: tuple[str, ...], value: str, location: str, fragment: str
) -> None:
    """Reject contradictory scales, non-time ODE clocks and nonlinear units.

    Parameters
    ----------
    path : tuple of str
        Nested package field to change.
    value : str
        Parseable but incompatible unit, or an unreadable profile-only unit.
    location, fragment : str
        Expected located refusal and authored explanation.
    """
    document = adex_candidate()
    target: dict[str, Any] = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    result = validate_candidate(document)
    assert not result.valid
    assert any(row.location == location and fragment in row.message for row in result.diagnostics)


def test_zero_state_and_physical_parameter_values_do_not_hide_bad_units() -> None:
    """Keep zero-valued physical quantities distinct from a literal zero."""
    document = adex_candidate()
    document["model"]["parameters"]["b_adapt"] = 0.0
    document["model"]["dynamics"]["v"] = "(v + w + b_adapt) / tau"
    assert not validate_candidate(document).valid


def test_named_dimensionless_zero_cannot_disguise_a_physical_assignment() -> None:
    """Reserve the additive-zero convention for literal expressions."""
    document = adex_candidate()
    document["model"]["parameters"]["zero"] = 0.0
    document["units"]["parameters"]["zero"] = "dimensionless"
    document["model"]["reset"]["v"] = "zero"
    assert not validate_candidate(document).valid


def test_bound_dt_parameter_retains_its_declared_unit_and_value() -> None:
    """Keep the actual profile's bound timebase parameter available to equations."""
    document = adex_candidate()
    document["model"]["parameters"]["dt"] = 0.1
    document["model"]["profile"]["parameters"]["dt"] = "timebase"
    document["model"]["profile"]["units"]["dt"] = "ms"
    document["units"]["parameters"]["dt"] = "ms"
    document["model"]["dynamics"]["v"] = "dt * v / (tau * tau)"
    assert validate_candidate(document).valid


def test_graded_output_keeps_its_physical_unit_without_changing_the_state_run() -> None:
    """Admit a physical scalar spike amplitude after checking its arithmetic."""
    document = adex_candidate()
    document["model"]["threshold"]["spike_value"] = "v - v_reset"
    assert validate_candidate(document).valid
    assert simulate_candidate(document, current=800.0, steps=3)["diverged_at_step"] is None


def test_extreme_declared_time_refuses_unrepresentable_derivative_and_rate_scales() -> None:
    """Locate physical scale overflow before numerical integration can start."""
    document = adex_candidate()
    document["units"]["time"] = "s * ms**107 / s**107"
    document["model"]["threshold"] = {
        "condition": "stochastic",
        "detection": "escape_rate",
        "rate_expression": "1 / tau",
    }
    result = validate_candidate(document)
    assert not result.valid
    locations = {
        row.location for row in result.diagnostics if "unit scale is not finite" in row.message
    }
    assert "/model/dynamics/v" in locations
    assert "/model/threshold/rate_expression" in locations


def test_explicitly_absent_reset_retains_the_existing_dsl_contract() -> None:
    """Treat a null reset like an omitted reset without inventing unit errors."""
    document = adex_candidate()
    document["model"]["reset"] = None
    assert validate_candidate(document).valid
    assert simulate_candidate(document, current=0.0, steps=3)["diverged_at_step"] is None


def test_shipped_fitting_candidate_has_coherent_units_and_the_same_numeric_drive() -> None:
    """Exercise the real imported fitting package with its unchanged input scale."""
    path = (
        Path(__file__).resolve().parents[1]
        / "studio/frontend/e2e/fixtures/LeakyIntegratorToFit.candidate.json"
    )
    document = json.loads(path.read_text())
    validation = validate_candidate(document)
    assert validation.valid, validation.diagnostics
    result = simulate_candidate(document, current=5.0, steps=3)
    assert result["trace"]["v"] == [-60.0, -55.5, -51.45]
    assert result["spike_count"] == 0


@pytest.mark.parametrize("detection", [[], {}])
def test_container_valued_stochastic_detection_is_an_authored_refusal(detection: object) -> None:
    """Keep malformed JSON detector values out of membership hashing failures.

    Parameters
    ----------
    detection : object
        JSON list or object in place of a detector name.
    """
    document = adex_candidate()
    document["model"]["threshold"] = {"condition": "stochastic", "detection": detection}
    validation = validate_candidate(document)
    assert not validation.valid
    assert validation.diagnostics[0].location == "/model/threshold/condition"
    assert validation.diagnostics[0].message == "unknown symbol(s) stochastic"


def _diffusion_candidate() -> dict[str, Any]:
    """Declare a seeded diffusion control under the actual numeric DSL contract."""
    document = adex_candidate()
    document["name"] = document["model"]["metadata"]["name"] = "DiffusionUnitControl"
    document["model"]["parameters"]["sigma"] = 0.5
    document["units"]["parameters"]["sigma"] = "mV / ms**0.5"
    document["model"]["profile"]["parameters"]["sigma"] = "source"
    document["model"]["profile"]["units"]["sigma"] = "mV / ms**0.5"
    document["model"]["dynamics"] = {"v": "sigma * xi", "w": "0"}
    document["model"]["threshold"]["rng_seed"] = 17
    document["source"] = {"citation": "The documented DSL xi=N(0,1)/sqrt(dt) contract"}
    document["reference_tests"] = [
        {
            "name": "seeded diffusion remains finite",
            "current": 0.0,
            "steps": 64,
            "expect": {"spike_count": {"max": 0}, "final_state": {"v": {"min": -100, "max": -20}}},
        }
    ]
    return document


def test_candidate_diffusion_units_preserve_the_actual_seeded_numeric_trace() -> None:
    """Admit inverse-root-time noise without normalising or drawing during validation."""
    document = _diffusion_candidate()
    original = copy.deepcopy(document)
    direct = UniversalNeuron.from_dict(document["model"])
    validation = validate_candidate(document)
    assert validation.valid, validation.diagnostics
    expected = []
    for _ in range(64):
        direct.step(I=0.0)
        expected.append(float(direct.state["v"]))
    result = simulate_candidate(document, current=0.0, steps=64)
    assert result["trace"]["v"] == expected
    assert result["final_state"] == direct.state
    packet = review_packet(document)
    assert packet["reference_tests_passed"] is True
    assert packet["candidate"] == original == document


@pytest.mark.parametrize("unit", ["mV", "mV / ms"])
def test_candidate_diffusion_coefficient_units_must_include_inverse_root_time(unit: str) -> None:
    """Reject wrong diffusion dimensions even when package and profile declarations agree.

    Parameters
    ----------
    unit : str
        Incorrect declared noise-amplitude unit.
    """
    document = _diffusion_candidate()
    document["units"]["parameters"]["sigma"] = unit
    document["model"]["profile"]["units"]["sigma"] = unit
    validation = validate_candidate(document)
    assert not validation.valid
    assert any(
        row.location == "/model/dynamics/v" and "dimensions" in row.message
        for row in validation.diagnostics
    )
