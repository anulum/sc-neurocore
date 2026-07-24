# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArithmetic from former test_c_fixed_emitter.py

"""Focused suite: TestArithmetic from former test_c_fixed_emitter.py."""

from __future__ import annotations

from tests.c_fixed_emitter_support import *  # noqa: F403


class TestArithmetic:
    def test_add_sub(self):
        expr, *_ = _c("a + b - 1.0", state={"a": "s->a", "b": "s->b"})
        assert "+" in expr and "-" in expr

    def test_multiply_uses_fxmul(self):
        expr, *_ = _c("a * b", state={"a": "s->a", "b": "s->b"})
        assert "fxmul(" in expr

    def test_div_by_constant_becomes_reciprocal_multiply(self):
        # 1/4 in Q8.8 → 64; division by a literal is a reciprocal fxmul, no sc_wrap
        expr, *_ = _c("a / 4.0", state={"a": "s->a"})
        assert "fxmul(" in expr and "64" in expr and "sc_wrap" not in expr

    def test_div_by_variable_uses_shift_divide(self):
        expr, *_ = _c("a / b", state={"a": "s->a", "b": "s->b"})
        assert "sc_wrap(" in expr and "<< 8" in expr and "/" in expr

    def test_unary_negate(self):
        expr, *_ = _c("-a", state={"a": "s->a"})
        assert "(-(" in expr

    def test_unary_plus_is_identity(self):
        expr, *_ = _c("+a", state={"a": "s->a"})
        assert "s->a" in expr

    def test_non_literal_modulo_raises(self):
        with pytest.raises(ValueError, match="Modulo divisor must be a positive numeric literal"):
            _c("a % b", state={"a": "s->a", "b": "s->b"})

    def test_unsupported_unaryop_raises(self):
        with pytest.raises(ValueError, match="Unsupported unary op"):
            _c("~a", state={"a": "s->a"})
