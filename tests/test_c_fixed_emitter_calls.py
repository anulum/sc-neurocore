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
    """Function calls lowered by the bit-true C/Rust expression emitter."""

    @pytest.mark.parametrize(
        "fn", ["exp", "log", "sqrt", "tanh", "cosh", "exprel", "sigmoid", "sin", "cos"]
    )
    def test_transcendental_emits_table(self, fn: str) -> None:
        """Each transcendental emits one table with its argument and index statements."""
        _e, stmts, tables, *_ = _c(f"{fn}(v)", state={"v": "s->v"})
        assert len(tables) == 1
        assert any("_arg" in s for s in stmts) and any("_idx" in s for s in stmts)

    def test_expit_aliases_sigmoid(self) -> None:
        """``expit`` reads the sigmoid table."""
        _e, _s, tables, *_ = _c("expit(v)", state={"v": "s->v"})
        assert any("sigmoid" in t for t in tables)

    def test_abs(self) -> None:
        """``abs`` becomes a sign select."""
        expr, *_ = _c("abs(v)", state={"v": "s->v"})
        assert "< 0" in expr

    def test_clip_three_args(self) -> None:
        """``clip`` raises to the lower bound first, then limits to the upper (numpy.clip order)."""
        expr, *_ = _c("clip(v, -1.0, 1.0)", state={"v": "s->v"})
        upper_select = expr.index("> ((int64_t)(256))")
        assert expr.index("< (-(((int64_t)(256))))") < upper_select
        assert expr.count("?") == 3  # the raised value is read by the comparison and the result

    def test_clip_wrong_arity_is_refused(self) -> None:
        # np.clip needs a_min and a_max; passing v through would differ from the model.
        """``clip`` without both bounds is refused, as np.clip refuses it."""
        with pytest.raises(ValueError, match="exactly 3"):
            _c("clip(v)", state={"v": "s->v"})
        with pytest.raises(ValueError, match="exactly 3"):
            _c("clip(v, 1.0)", state={"v": "s->v"})

    def test_max(self) -> None:
        """Two-argument max becomes a greater-than select."""
        expr, *_ = _c("max(v, 0.0)", state={"v": "s->v"})
        assert ">" in expr

    def test_min(self) -> None:
        """Two-argument min becomes a less-than select."""
        expr, *_ = _c("min(v, 0.0)", state={"v": "s->v"})
        assert "<" in expr

    def test_max_single_arg_is_refused(self) -> None:
        # Python's max(x) on a number raises; the kernel must not invent a value.
        """``max`` with one argument is refused, as Python's max refuses a number."""
        with pytest.raises(ValueError, match="at least 2"):
            _c("max(v)", state={"v": "s->v"})

    def test_max_and_min_use_every_argument(self) -> None:
        # Folded left: max(max(v, 0), 1). Values are checked by the co-simulation tests.
        """``max`` and min fold left, so the last argument is the final select's operand."""
        expr, *_ = _c("max(v, 0.0, 1.0)", state={"v": "s->v"})
        assert expr.endswith(f"(((int64_t)({Q.encode(1.0)}))))")
        expr, *_ = _c("min(v, 0.0, 1.0, 2.0)", state={"v": "s->v"})
        assert expr.endswith(f"(((int64_t)({Q.encode(2.0)}))))")

    def test_keyword_arguments_are_refused(self) -> None:
        """Keyword arguments are refused rather than dropped."""
        with pytest.raises(ValueError, match="positional arguments only"):
            _c("clip(v, a_min=0.0, a_max=1.0)", state={"v": "s->v"})

    def test_pi_is_encoded_as_a_literal(self) -> None:
        """``pi`` is encoded as its signed Q-format integer."""
        expr, *_ = _c("2.0 * pi", state={})
        assert str(signed_q(Q, 3.141592653589793)) in expr

    def test_a_parameter_named_pi_takes_precedence(self) -> None:
        """A parameter named pi is read instead of the constant."""
        expr, *_ = _c("pi", state={}, params={"pi": 7})
        assert "7" in expr and str(signed_q(Q, 3.141592653589793)) not in expr

    def test_unsupported_function_raises(self) -> None:
        """A function outside the vocabulary is refused by name."""
        with pytest.raises(ValueError, match="Unsupported function"):
            _c("gamma(v)", state={"v": "s->v"})

    def test_non_name_callable_raises(self) -> None:
        """An attribute call such as m.f(v) is refused."""
        with pytest.raises(ValueError, match="Only named function calls"):
            _c("m.f(v)", state={"v": "s->v"})

    def test_no_args_raises(self) -> None:
        """A call without arguments is refused."""
        with pytest.raises(ValueError, match="requires at least 1 argument"):
            _c("exp()")
