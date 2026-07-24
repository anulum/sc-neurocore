# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParseHelpers from former test_neuroml_import.py

"""Focused suite: TestParseHelpers from former test_neuroml_import.py."""

from __future__ import annotations

from tests.neuroml_import_support import *  # noqa: F403


class TestParseHelpers:
    def test_parse_unit_value_none_is_zero(self):
        assert _parse_unit_value(None) == 0.0

    def test_parse_unit_value_dimensionless_falls_through(self):
        # No recognised unit suffix -> parsed as a bare float.
        assert _parse_unit_value("0.7") == pytest.approx(0.7)

    def test_parse_current_pa_none_is_zero(self):
        assert _parse_current_pa(None) == 0.0

    def test_parse_current_pa_dimensionless_falls_through(self):
        assert _parse_current_pa("42") == pytest.approx(42.0)
