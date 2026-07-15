# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for DSL schema validator error and parity edges

"""Contracts for schema-validator error reporting, file lookup and TOML/JSON parity."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.neurons import schema_validator
from sc_neurocore.neurons.schema_validator import (
    SchemaError,
    validate_schema,
    validate_schema_dict,
)


def _valid_schema() -> dict[str, Any]:
    """A minimal schema dictionary that passes structural validation."""
    return {
        "metadata": {"schema_version": 1, "name": "demo"},
        "state": {"v": 0.0},
        "dynamics": {"v": "v + I"},
        "integration": {"dt": 0.1, "method": "euler"},
    }


def _levels(errors: list[SchemaError], message_fragment: str) -> list[SchemaError]:
    """Return the errors whose message contains the given fragment."""
    return [e for e in errors if message_fragment in e.message]


def test_schema_error_repr_includes_section_prefix() -> None:
    """SchemaError.__repr__ prefixes the section in brackets and upper-cases the level."""
    assert repr(SchemaError("error", "boom", "metadata")) == "ERROR: [metadata] boom"
    assert repr(SchemaError("warning", "soft")) == "WARNING: soft"


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
    schema["state"]["w"] = 0.0  # no matching dynamics entry

    errors = validate_schema_dict(schema, "demo")

    warnings = _levels(errors, "has no dynamics equation")
    assert warnings
    assert all(w.level == "warning" for w in warnings)


def test_unknown_top_level_section_warns() -> None:
    """An unrecognised top-level section produces a warning."""
    schema = _valid_schema()
    schema["telemetry"] = {"foo": 1}

    errors = validate_schema_dict(schema, "demo")

    assert _levels(errors, "Unknown section 'telemetry'")


def _write_schema_files(
    directory: Path, name: str, *, toml: str | None, data: dict[str, Any] | None
) -> None:
    """Write optional TOML and JSON variants of a schema into a directory."""
    if toml is not None:
        (directory / f"{name}.toml").write_text(toml, encoding="utf-8")
    if data is not None:
        (directory / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def test_validate_schema_reports_no_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A name with neither TOML nor JSON present yields a single not-found error."""
    monkeypatch.setattr(schema_validator, "_SCHEMAS_DIR", tmp_path)

    errors = validate_schema("ghost")

    assert _levels(errors, "No schema files found for 'ghost'")


def test_validate_schema_flags_toml_json_key_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When both formats exist but their state keys differ, parity reports a mismatch."""
    monkeypatch.setattr(schema_validator, "_SCHEMAS_DIR", tmp_path)
    toml_text = (
        '[metadata]\nschema_version = 1\nname = "mm"\n'
        "[state]\nv = 0.0\n"
        '[dynamics]\nv = "v + I"\n'
        '[integration]\ndt = 0.1\nmethod = "euler"\n'
    )
    json_data = {
        "metadata": {"schema_version": 1, "name": "mm"},
        "state": {"w": 0.0},
        "dynamics": {"w": "w + I"},
        "integration": {"dt": 0.1, "method": "euler"},
    }
    _write_schema_files(tmp_path, "mm", toml=toml_text, data=json_data)

    errors = validate_schema("mm")

    assert _levels(errors, "TOML/JSON key mismatch in 'state'")


def test_validate_schema_flags_toml_json_value_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Equal section keys cannot hide a different authored timestep value."""
    monkeypatch.setattr(schema_validator, "_SCHEMAS_DIR", tmp_path)
    toml_text = (
        '[metadata]\nschema_version = 1\nname = "mm"\n'
        "[state]\nv = 0.0\n"
        '[dynamics]\nv = "v + I"\n'
        '[integration]\ndt = 0.1\nmethod = "euler"\n'
    )
    json_data = _valid_schema()
    json_data["metadata"]["name"] = "mm"
    json_data["integration"]["dt"] = 0.2
    _write_schema_files(tmp_path, "mm", toml=toml_text, data=json_data)

    errors = validate_schema("mm")

    assert _levels(errors, "TOML/JSON value mismatch in 'integration'")


def test_validate_schema_flags_v2_knowledge_layer_value_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Version 2 authored knowledge cannot drift between TOML and JSON."""
    monkeypatch.setattr(schema_validator, "_SCHEMAS_DIR", tmp_path)
    toml_text = (
        '[metadata]\nschema_version = 2\nname = "v2-mm"\n'
        "[state]\nv = 0.0\n"
        '[dynamics]\nv = "v + I"\n'
        '[integration]\ndt = 0.1\nmethod = "euler"\n'
        '[science]\nequations_as_published = "dv/dt = v + I"\n'
    )
    json_data = _valid_schema()
    json_data["metadata"] = {"schema_version": 2, "name": "v2-mm"}
    json_data["science"] = {"equations_as_published": "dv/dt = -v + I"}
    _write_schema_files(tmp_path, "v2-mm", toml=toml_text, data=json_data)

    errors = validate_schema("v2-mm")

    assert _levels(errors, "TOML/JSON value mismatch in 'science'")


def test_validate_schema_reports_malformed_v2_layer_without_parity_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Paired non-object v2 layers report structural errors without crashing."""
    monkeypatch.setattr(schema_validator, "_SCHEMAS_DIR", tmp_path)
    toml_text = (
        'science = ["not-an-object"]\n'
        '[metadata]\nschema_version = 2\nname = "v2-bad"\n'
        "[state]\nv = 0.0\n"
        '[dynamics]\nv = "v + I"\n'
        '[integration]\ndt = 0.1\nmethod = "euler"\n'
    )
    json_data = _valid_schema()
    json_data["metadata"] = {"schema_version": 2, "name": "v2-bad"}
    json_data["science"] = ["not-an-object"]
    _write_schema_files(tmp_path, "v2-bad", toml=toml_text, data=json_data)

    errors = validate_schema("v2-bad")

    assert len(_levels(errors, "Section 'science' must be an object")) == 2


def test_validate_schema_warns_when_json_variant_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A TOML-only schema warns that the JSON variant is missing and exercises the tomli path."""
    monkeypatch.setattr(schema_validator, "_SCHEMAS_DIR", tmp_path)
    # Force the tomli fallback import branch by hiding the stdlib tomllib module.
    monkeypatch.setitem(sys.modules, "tomllib", None)
    toml_text = (
        '[metadata]\nschema_version = 1\nname = "tonly"\n'
        "[state]\nv = 0.0\n"
        '[dynamics]\nv = "v + I"\n'
        '[integration]\ndt = 0.1\nmethod = "euler"\n'
    )
    _write_schema_files(tmp_path, "tonly", toml=toml_text, data=None)

    errors = validate_schema("tonly")

    assert _levels(errors, "Missing JSON version for 'tonly'")


def test_validate_schema_warns_when_toml_variant_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A JSON-only schema warns that the TOML variant is missing."""
    monkeypatch.setattr(schema_validator, "_SCHEMAS_DIR", tmp_path)
    json_data = _valid_schema()
    json_data["metadata"]["name"] = "jonly"
    _write_schema_files(tmp_path, "jonly", toml=None, data=json_data)

    errors = validate_schema("jonly")

    assert _levels(errors, "Missing TOML version for 'jonly'")
