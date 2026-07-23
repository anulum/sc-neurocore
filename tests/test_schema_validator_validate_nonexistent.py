# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidateNonexistent from former test_schema_validator.py

"""Focused suite: TestValidateNonexistent from former test_schema_validator.py."""

from __future__ import annotations

from tests.schema_validator_support import *  # noqa: F403

class TestValidateNonexistent:
    """Test error handling for missing schemas."""

    def test_nonexistent_schema(self) -> None:
        errors = validate_schema("totally_fake_model")
        assert len(errors) == 1
        assert "No schema files found" in errors[0].message
