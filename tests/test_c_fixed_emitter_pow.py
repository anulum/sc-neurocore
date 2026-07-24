# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPow from former test_c_fixed_emitter.py

"""Focused suite: TestPow from former test_c_fixed_emitter.py."""

from __future__ import annotations

from tests.c_fixed_emitter_support import *  # noqa: F403


class TestPow:
    @pytest.mark.parametrize("exp", [2, 3, 4, 5, 8])
    def test_integer_powers_expand_to_fxmul_chain(self, exp):
        expr, *_ = _c(f"a ** {exp}", state={"a": "s->a"})
        assert expr.count("fxmul(") == exp - 1

    def test_sqrt_power_uses_lut(self):
        expr, stmts, tables, *_ = _c("a ** 0.5", state={"a": "s->a"})
        assert any("sqrt" in t for t in tables)

    def test_cbrt_power_uses_lut(self):
        expr, stmts, tables, *_ = _c("a ** (1.0/3.0)", state={"a": "s->a"})
        assert any("cbrt" in t for t in tables)

    def test_unsupported_power_raises(self):
        with pytest.raises(ValueError, match="Only integer powers"):
            _c("a ** 1.7", state={"a": "s->a"})
