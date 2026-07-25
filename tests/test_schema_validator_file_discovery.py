# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Schema validator file discovery tests

"""Validate missing schema files and one-sided TOML or JSON variants."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sc_neurocore.neurons import schema_validator
from sc_neurocore.neurons.schema_validator import validate_schema
from tests.schema_validator_contract_support import (
    _levels,
    _valid_schema,
    _write_schema_files,
)


def test_validate_schema_reports_no_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A name with neither TOML nor JSON present yields a single not-found error."""
    monkeypatch.setattr(schema_validator, "_SCHEMAS_DIR", tmp_path)

    errors = validate_schema("ghost")

    assert _levels(errors, "No schema files found for 'ghost'")


def test_validate_schema_warns_when_json_variant_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A TOML-only schema warns that the JSON variant is missing and exercises tomli."""
    monkeypatch.setattr(schema_validator, "_SCHEMAS_DIR", tmp_path)
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A JSON-only schema warns that the TOML variant is missing."""
    monkeypatch.setattr(schema_validator, "_SCHEMAS_DIR", tmp_path)
    json_data = _valid_schema()
    json_data["metadata"]["name"] = "jonly"
    _write_schema_files(tmp_path, "jonly", toml=None, data=json_data)

    errors = validate_schema("jonly")

    assert _levels(errors, "Missing TOML version for 'jonly'")
