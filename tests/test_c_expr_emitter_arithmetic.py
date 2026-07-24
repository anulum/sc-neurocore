# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArithmetic from former test_c_expr_emitter.py

"""Focused suite: TestArithmetic from former test_c_expr_emitter.py."""

from __future__ import annotations

from tests.c_expr_emitter_support import *  # noqa: F403


class TestArithmetic:
    def test_add_sub_mul_div(self) -> None:
        assert _emit("a + b") == "(a + b)"
        assert _emit("a - b") == "(a - b)"
        assert _emit("a * b") == "(a * b)"
        assert _emit("a / b") == "(a / b)"

    def test_unary_minus(self) -> None:
        assert _emit("-a") == "(-a)"

    def test_unary_plus_passthrough(self) -> None:
        assert _emit("+a") == "a"

    def test_constant_cast_to_fp_type(self) -> None:
        assert _emit("1.5") == "fp_t(1.5)"

    def test_nested(self) -> None:
        assert _emit("(a + b) * c") == "((a + b) * c)"
