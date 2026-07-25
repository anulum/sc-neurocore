# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Schema validator TOML and JSON parity tests

"""Validate semantic parity between authored TOML and JSON schema variants."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.neurons import schema_validator
from sc_neurocore.neurons.schema_validator import validate_schema
from tests.schema_validator_contract_support import (
    _levels,
    _valid_schema,
    _write_schema_files,
)


def test_validate_schema_flags_toml_json_key_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
