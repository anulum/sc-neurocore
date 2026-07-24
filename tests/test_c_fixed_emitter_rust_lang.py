# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustLang from former test_c_fixed_emitter.py

"""Focused suite: TestRustLang from former test_c_fixed_emitter.py."""

from __future__ import annotations

from tests.c_fixed_emitter_support import *  # noqa: F403


class TestRustLang:
    def test_rust_cast_syntax(self):
        expr, *_ = emit_c_fixed_expr("a + 1.0", {"a": "self.a"}, {}, Q, lang="rust")
        assert "as i64" in expr

    def test_rust_conditional_expression(self):
        expr, *_ = emit_c_fixed_expr("abs(v)", {"v": "self.v"}, {}, Q, lang="rust")
        assert "if (" in expr and "else" in expr

    def test_rust_lut_index_cast(self):
        _e, stmts, *_ = emit_c_fixed_expr("exp(v)", {"v": "self.v"}, {}, Q, lang="rust")
        assert any("let " in s and ": i64" in s for s in stmts)
        assert "as usize" in _e
