# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore — Tests for the DSL schema validator

"""Test suite for schema_validator.py — ensures all bundled schemas pass validation."""

from __future__ import annotations


from sc_neurocore.neurons.schema_validator import (
    validate_all_bundled,
    validate_schema,
    validate_schema_dict,
)


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


class TestValidateBundledSchemas:
    """Validate all 12 bundled schemas."""

    def test_all_bundled_schemas_have_no_errors(self) -> None:
        results = validate_all_bundled()
        assert len(results) >= 12, f"Expected ≥12 schemas, found {len(results)}"

        for name, errors in results.items():
            real_errors = [e for e in errors if e.level == "error"]
            assert len(real_errors) == 0, f"Schema '{name}' has errors: {real_errors}"

    def test_all_bundled_schemas_have_both_formats(self) -> None:
        results = validate_all_bundled()
        for name, errors in results.items():
            missing_format = [
                e
                for e in errors
                if e.level == "warning"
                and "Missing" in e.message
                and ("JSON version" in e.message or "TOML version" in e.message)
            ]
            assert len(missing_format) == 0, f"Schema '{name}' missing a format: {missing_format}"

    def test_lif_validates_clean(self) -> None:
        errors = validate_schema("lif")
        real_errors = [e for e in errors if e.level == "error"]
        assert len(real_errors) == 0

    def test_hodgkin_huxley_validates_clean(self) -> None:
        errors = validate_schema("hodgkin_huxley")
        real_errors = [e for e in errors if e.level == "error"]
        assert len(real_errors) == 0

    def test_glif_validates_clean(self) -> None:
        errors = validate_schema("glif")
        real_errors = [e for e in errors if e.level == "error"]
        assert len(real_errors) == 0


class TestValidateNonexistent:
    """Test error handling for missing schemas."""

    def test_nonexistent_schema(self) -> None:
        errors = validate_schema("totally_fake_model")
        assert len(errors) == 1
        assert "No schema files found" in errors[0].message
