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

    def test_clip_wrong_arity_is_refused(self):
        # np.clip needs a_min and a_max; passing v through would differ from the model.
        with pytest.raises(ValueError, match="exactly 3"):
            _c("clip(v)", state={"v": "s->v"})
        with pytest.raises(ValueError, match="exactly 3"):
            _c("clip(v, 1.0)", state={"v": "s->v"})

    def test_max(self):
        expr, *_ = _c("max(v, 0.0)", state={"v": "s->v"})
        assert ">" in expr

    def test_min(self):
        expr, *_ = _c("min(v, 0.0)", state={"v": "s->v"})
        assert "<" in expr

    def test_max_single_arg_is_refused(self):
        # Python's max(x) on a number raises; the kernel must not invent a value.
        with pytest.raises(ValueError, match="at least 2"):
            _c("max(v)", state={"v": "s->v"})

    def test_max_and_min_use_every_argument(self):
        # Folded left: max(max(v, 0), 1). Values are checked by the co-simulation tests.
        expr, *_ = _c("max(v, 0.0, 1.0)", state={"v": "s->v"})
        assert expr.endswith(f"(((int64_t)({Q.encode(1.0)}))))")
        expr, *_ = _c("min(v, 0.0, 1.0, 2.0)", state={"v": "s->v"})
        assert expr.endswith(f"(((int64_t)({Q.encode(2.0)}))))")

    def test_keyword_arguments_are_refused(self):
        with pytest.raises(ValueError, match="positional arguments only"):
            _c("clip(v, a_min=0.0, a_max=1.0)", state={"v": "s->v"})

    def test_pi_is_encoded_as_a_literal(self):
        expr, *_ = _c("2.0 * pi", state={})
        assert str(signed_q(Q, 3.141592653589793)) in expr

    def test_a_parameter_named_pi_takes_precedence(self):
        expr, *_ = _c("pi", state={}, params={"pi": 7})
        assert "7" in expr and str(signed_q(Q, 3.141592653589793)) not in expr

    def test_unsupported_function_raises(self):
        with pytest.raises(ValueError, match="Unsupported function"):
            _c("gamma(v)", state={"v": "s->v"})

    def test_non_name_callable_raises(self):
        with pytest.raises(ValueError, match="Only named function calls"):
            _c("m.f(v)", state={"v": "s->v"})

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="requires at least 1 argument"):
            _c("exp()")
