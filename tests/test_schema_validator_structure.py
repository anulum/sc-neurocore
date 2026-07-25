# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Schema validator structural contract tests

"""Validate required schema fields, event-only contracts, and warnings."""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.neurons.schema_validator import validate_schema_dict
from tests.schema_validator_contract_support import _levels, _valid_schema


def test_missing_metadata_field_is_reported() -> None:
    """A metadata block without the required name field yields a metadata error."""
    schema = _valid_schema()
    del schema["metadata"]["name"]

    errors = validate_schema_dict(schema, "demo")

    assert _levels(errors, "Missing metadata field 'name'")


def test_non_numeric_state_initial_value_is_reported() -> None:
    """A state variable whose initial value is not numeric is rejected."""
    schema = _valid_schema()
    schema["state"]["v"] = "not-a-number"

    errors = validate_schema_dict(schema, "demo")

    assert _levels(errors, "non-numeric initial value")


def test_missing_integration_field_is_reported() -> None:
    """An integration block missing the method field is rejected."""
    schema = _valid_schema()
    del schema["integration"]["method"]

    errors = validate_schema_dict(schema, "demo")

    assert _levels(errors, "Missing integration field 'method'")


def test_empty_dynamics_section_is_reported() -> None:
    """An empty dynamics section is flagged as an error."""
    schema = _valid_schema()
    schema["dynamics"] = {}

    errors = validate_schema_dict(schema, "demo")

    assert _levels(errors, "Dynamics section is empty")


@pytest.mark.parametrize(
    "threshold",
    [
        {"condition": "I >= theta", "detection": "level"},
        {
            "condition": "stochastic",
            "detection": "poisson",
            "probability_expression": "1.0 - exp(-rate_hz * dt_ms / 1000.0)",
        },
    ],
)
def test_supported_event_only_schemas_match_runtime_contract(
    threshold: dict[str, str],
) -> None:
    """Validator and runtime accept the same exact state-free event contracts."""
    from sc_neurocore.neurons.universal_dsl import UniversalNeuron

    schema: dict[str, Any] = {
        "metadata": {"schema_version": 1, "name": "event-only"},
        "state": {},
        "parameters": {"theta": 1.0, "rate_hz": 100.0, "dt_ms": 1.0},
        "dynamics": {},
        "integration": {"dt": 1.0, "method": "euler"},
        "threshold": threshold,
    }

    errors = validate_schema_dict(schema, "event-only")

    assert not [error for error in errors if error.level == "error"]
    UniversalNeuron(schema)


@pytest.mark.parametrize(
    "threshold",
    [
        {"condition": "", "detection": "level"},
        {"condition": "stochastic", "detection": "poisson"},
        {
            "condition": "stochastic",
            "detection": "poisson",
            "probability_expression": "   ",
        },
    ],
)
def test_malformed_event_only_schema_still_requires_state_and_dynamics(
    threshold: dict[str, str],
) -> None:
    """Labels alone never waive the ordinary state and dynamics requirements."""
    schema = _valid_schema()
    schema["state"] = {}
    schema["dynamics"] = {}
    schema["threshold"] = threshold

    errors = validate_schema_dict(schema, "malformed-event-only")

    assert _levels(errors, "State section is empty")
    assert _levels(errors, "Dynamics section is empty")


def test_state_variable_without_dynamics_equation_warns() -> None:
    """A state variable that has no dynamics equation produces a warning."""
    schema = _valid_schema()
    schema["state"]["w"] = 0.0

    errors = validate_schema_dict(schema, "demo")

    warnings = _levels(errors, "has no dynamics equation")
    assert warnings
    assert all(warning.level == "warning" for warning in warnings)


def test_unknown_top_level_section_warns() -> None:
    """An unrecognised top-level section produces a warning."""
    schema = _valid_schema()
    schema["telemetry"] = {"foo": 1}

    errors = validate_schema_dict(schema, "demo")

    assert _levels(errors, "Unknown section 'telemetry'")
