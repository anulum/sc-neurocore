# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConstFloat from former test_expr_lut_tables.py

"""Focused suite: TestConstFloat from former test_expr_lut_tables.py."""

from __future__ import annotations

from tests.expr_lut_tables_support import *  # noqa: F403


class TestConstFloat:
    def _fold(self, expr: str) -> float | None:
        return tables.const_float(ast.parse(expr, mode="eval").body)

    def test_literal(self) -> None:
        assert self._fold("3.5") == 3.5

    def test_unary_minus(self) -> None:
        assert self._fold("-2.0") == -2.0

    def test_division(self) -> None:
        assert self._fold("1.0 / 3.0") == 1.0 / 3.0

    def test_mult_add_sub(self) -> None:
        assert self._fold("2 * 3") == 6.0
        assert self._fold("2 + 3") == 5.0
        assert self._fold("5 - 3") == 2.0

    def test_division_by_zero_is_none(self) -> None:
        assert self._fold("1.0 / 0.0") is None

    def test_non_constant_is_none(self) -> None:
        assert self._fold("x + 1") is None

    def test_nested_non_constant_is_none(self) -> None:
        assert self._fold("x * 2 + 1") is None
