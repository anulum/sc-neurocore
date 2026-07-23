# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCalls from former test_c_fixed_emitter.py

"""Focused suite: TestCalls from former test_c_fixed_emitter.py."""

from __future__ import annotations

from tests.c_fixed_emitter_support import *  # noqa: F403

class TestCalls:
    @pytest.mark.parametrize(
        "fn", ["exp", "log", "sqrt", "tanh", "cosh", "exprel", "sigmoid", "sin", "cos"]
    )
    def test_transcendental_emits_table(self, fn):
        _e, stmts, tables, *_ = _c(f"{fn}(v)", state={"v": "s->v"})
        assert len(tables) == 1
        assert any("_arg" in s for s in stmts) and any("_idx" in s for s in stmts)

    def test_expit_aliases_sigmoid(self):
        _e, _s, tables, *_ = _c("expit(v)", state={"v": "s->v"})
        assert any("sigmoid" in t for t in tables)

    def test_abs(self):
        expr, *_ = _c("abs(v)", state={"v": "s->v"})
        assert "< 0" in expr

    def test_clip_three_args(self):
        expr, *_ = _c("clip(v, -1.0, 1.0)", state={"v": "s->v"})
        assert expr.count("?") == 2 or "if (" in expr

    def test_clip_wrong_arity_is_identity(self):
        expr, *_ = _c("clip(v)", state={"v": "s->v"})
        assert "s->v" in expr and "?" not in expr

    def test_max(self):
        expr, *_ = _c("max(v, 0.0)", state={"v": "s->v"})
        assert ">" in expr

    def test_min(self):
        expr, *_ = _c("min(v, 0.0)", state={"v": "s->v"})
        assert "<" in expr

    def test_max_single_arg_is_identity(self):
        expr, *_ = _c("max(v)", state={"v": "s->v"})
        assert "s->v" in expr

    def test_unsupported_function_raises(self):
        with pytest.raises(ValueError, match="Unsupported function"):
            _c("gamma(v)", state={"v": "s->v"})

    def test_non_name_callable_raises(self):
        with pytest.raises(ValueError, match="Only named function calls"):
            _c("m.f(v)", state={"v": "s->v"})

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="requires at least 1 argument"):
            _c("exp()")
