# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidateBundledSchemas from former test_schema_validator.py

"""Focused suite: TestValidateBundledSchemas from former test_schema_validator.py."""

from __future__ import annotations

from tests.schema_validator_support import *  # noqa: F403


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

    def test_wong_wang_v2_validates_without_version_or_layer_warnings(self) -> None:
        """The paired Wong-Wang v2 schema passes the complete static validator."""
        errors = validate_schema("wong_wang")

        assert not [error for error in errors if error.level == "error"]
        assert not [error for error in errors if "Unknown section" in error.message]
