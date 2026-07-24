# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSupportedFunctions from former test_expr_lut_tables.py

"""Focused suite: TestSupportedFunctions from former test_expr_lut_tables.py."""

from __future__ import annotations

from tests.expr_lut_tables_support import *  # noqa: F403


class TestSupportedFunctions:
    def test_contains_transcendentals_and_helpers(self) -> None:
        assert {"exp", "log", "sqrt", "tanh", "cosh", "exprel", "sin", "cos"} <= (
            tables.SUPPORTED_FUNCTIONS
        )
        assert {"sigmoid", "expit", "abs", "clip", "max", "min"} <= tables.SUPPORTED_FUNCTIONS

    def test_is_frozen(self) -> None:
        assert isinstance(tables.SUPPORTED_FUNCTIONS, frozenset)
