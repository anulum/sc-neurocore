# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidateSchemaDict from former test_schema_validator.py

"""Focused suite: TestValidateSchemaDict from former test_schema_validator.py."""

from __future__ import annotations

from tests.schema_validator_support import *  # noqa: F403

class TestValidateSchemaDict:
    """Test the core validation logic."""

    def test_valid_minimal_schema(self) -> None:
        schema = {
            "metadata": {"schema_version": 1, "name": "Test"},
            "state": {"v": 0.0},
            "parameters": {},
            "integration": {"dt": 0.01, "method": "euler"},
            "dynamics": {"v": "v + I"},
        }
        errors = validate_schema_dict(schema, "test")
        real_errors = [e for e in errors if e.level == "error"]
        assert len(real_errors) == 0

    def test_valid_v2_schema_recognises_all_knowledge_layers(self) -> None:
        """Version 2 knowledge layers are supported structured sections."""
        schema = {
            "metadata": {"schema_version": 2, "name": "Test v2"},
            "state": {"v": 0.0},
            "parameters": {},
            "integration": {"dt": 0.01, "method": "euler"},
            "dynamics": {"v": "v + I"},
            "science": {"equations_as_published": "dv/dt = v + I"},
            "validation": {"metric": "trajectory"},
            "provenance": {"contributors": ["A. Researcher"]},
            "hints": {"recommended_precision": "Q16.16"},
        }

        errors = validate_schema_dict(schema, "test-v2")

        assert not [error for error in errors if error.level == "error"]
        assert not [error for error in errors if "Unknown section" in error.message]

    def test_v2_knowledge_layers_must_be_objects(self) -> None:
        """Malformed version 2 knowledge layers fail before runtime access."""
        schema = {
            "metadata": {"schema_version": 2, "name": "Test v2"},
            "state": {"v": 0.0},
            "parameters": {},
            "integration": {"dt": 0.01, "method": "euler"},
            "dynamics": {"v": "v + I"},
            "science": ["not", "an", "object"],
        }

        errors = validate_schema_dict(schema, "test-v2")

        assert any("Section 'science' must be an object" in error.message for error in errors)

    def test_missing_required_section(self) -> None:
        schema = {
            "metadata": {"schema_version": 1, "name": "Test"},
            "state": {"v": 0.0},
        }
        errors = validate_schema_dict(schema, "test")
        assert any("Missing required section" in e.message for e in errors)

    def test_unsupported_version(self) -> None:
        schema = {
            "metadata": {"schema_version": 99, "name": "Test"},
            "state": {"v": 0.0},
            "parameters": {},
            "integration": {"dt": 0.01, "method": "euler"},
            "dynamics": {"v": "v + I"},
        }
        errors = validate_schema_dict(schema, "test")
        assert any("Unsupported schema version" in e.message for e in errors)

    def test_empty_state_is_error(self) -> None:
        schema = {
            "metadata": {"schema_version": 1, "name": "Test"},
            "state": {},
            "parameters": {},
            "integration": {"dt": 0.01, "method": "euler"},
            "dynamics": {"v": "v + I"},
        }
        errors = validate_schema_dict(schema, "test")
        assert any("State section is empty" in e.message for e in errors)

    def test_negative_dt_is_error(self) -> None:
        schema = {
            "metadata": {"schema_version": 1, "name": "Test"},
            "state": {"v": 0.0},
            "parameters": {},
            "integration": {"dt": -0.01, "method": "euler"},
            "dynamics": {"v": "v + I"},
        }
        errors = validate_schema_dict(schema, "test")
        assert any("dt must be positive" in e.message for e in errors)

    def test_warns_on_missing_metadata(self) -> None:
        schema = {
            "metadata": {"schema_version": 1, "name": "Test"},
            "state": {"v": 0.0},
            "parameters": {},
            "integration": {"dt": 0.01, "method": "euler"},
            "dynamics": {"v": "v + I"},
        }
        errors = validate_schema_dict(schema, "test")
        warnings = [e for e in errors if e.level == "warning"]
        # Should warn about missing author, year, doi, description
        assert any("Missing recommended" in e.message for e in warnings)
