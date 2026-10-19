# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFunctionCalls from former test_c_expr_emitter.py

"""Focused suite: TestFunctionCalls from former test_c_expr_emitter.py."""

from __future__ import annotations

from tests.c_expr_emitter_support import *  # noqa: F403


class TestFunctionCalls:
    """Function calls lowered by the HLS C++ expression emitter."""

    def test_direct_math_functions(self) -> None:
        """Transcendentals map to the math namespace's function of the same name."""
        assert _emit("exp(v)", {"v"}) == "hls::exp(v)"
        assert _emit("tanh(v)", {"v"}) == "hls::tanh(v)"
        assert _emit("cosh(v)", {"v"}) == "hls::cosh(v)"
        assert _emit("sin(v)", {"v"}) == "hls::sin(v)"
        assert _emit("log(v)", {"v"}) == "hls::log(v)"

    def test_std_namespace_option(self) -> None:
        """The math namespace can be switched from hls to std."""
        assert _emit("exp(v)", {"v"}, math_ns="std") == "std::exp(v)"

    def test_sigmoid_and_expit_use_helper(self) -> None:
        """``sigmoid`` and its alias expit call the inline sc_sigmoid helper."""
        assert _emit("sigmoid(v)", {"v"}) == "sc_sigmoid(v)"
        assert _emit("expit(v)", {"v"}) == "sc_sigmoid(v)"

    def test_exprel_uses_helper(self) -> None:
        """``exprel`` calls the inline sc_exprel helper."""
        assert _emit("exprel(v)", {"v"}) == "sc_exprel(v)"

    def test_abs(self) -> None:
        """``abs`` maps to the math namespace's abs."""
        assert _emit("abs(v)", {"v"}) == "hls::abs(v)"

    def test_clip_three_args(self) -> None:
        """``clip`` with both bounds becomes two selects against the typed bounds."""
        out = _emit("clip(v, 0.0, 1.0)", {"v"})
        assert "? fp_t(0.0)" in out and "? fp_t(1.0)" in out

    def test_max_min(self) -> None:
        """Two-argument max and min become one select each."""
        assert _emit("max(a, b)") == "((a > b) ? a : b)"
        assert _emit("min(a, b)") == "((a < b) ? a : b)"

    def test_clip_wrong_arity_is_refused(self) -> None:
        # np.clip needs both bounds; passing v through would differ from the model.
        """``clip`` without both bounds is refused, as np.clip refuses it."""
        with pytest.raises(ValueError, match="exactly 3"):
            _emit("clip(v)", {"v"})
        with pytest.raises(ValueError, match="exactly 3"):
            _emit("clip(v, 1.0)", {"v"})

    def test_max_single_arg_is_refused(self) -> None:
        """``max`` with one argument is refused, as Python's max refuses a number."""
        with pytest.raises(ValueError, match="at least 2"):
            _emit("max(v)", {"v"})

    def test_max_min_use_every_argument(self) -> None:
        """``max`` and min with three arguments fold left over all of them."""
        assert _emit("max(a, b, c)") == "((((a > b) ? a : b) > c) ? ((a > b) ? a : b) : c)"
        assert _emit("min(a, b, c)") == "((((a < b) ? a : b) < c) ? ((a < b) ? a : b) : c)"

    def test_keyword_arguments_are_refused(self) -> None:
        """Keyword arguments are refused rather than dropped."""
        with pytest.raises(ValueError, match="positional arguments only"):
            _emit("clip(v, a_min=0.0, a_max=1.0)", {"v"})

    def test_sinh_is_a_direct_math_call(self) -> None:
        """``sinh`` lowers to the math namespace's ``sinh``, not to its argument."""
        assert _emit("sinh(v)", {"v"}) == "hls::sinh(v)"

    def test_every_supported_function_has_a_lowering(self) -> None:
        """No vocabulary entry falls through to another function's lowering; draws refuse."""
        from sc_neurocore.compiler.expr_lut_tables import FUNCTION_ARITY, RANDOM_FUNCTIONS

        for name, (count, _maximum) in FUNCTION_ARITY.items():
            call = f"{name}({', '.join(['v'] * count)})"
            if name in RANDOM_FUNCTIONS:
                with pytest.raises(ValueError, match="no C/C\\+\\+ \\(HLS\\) lowering"):
                    _emit(call, {"v"})
                continue
            assert _emit(call, {"v"}) != "v", name

    def test_pi_is_a_typed_literal(self) -> None:
        """``pi`` becomes a literal of the fixed-point type."""
        assert _emit("pi") == "fp_t(3.141592653589793)"

    def test_unsupported_function_raises(self) -> None:
        """A function outside the vocabulary is refused by name."""
        with pytest.raises(ValueError, match="Unsupported function"):
            _emit("gamma(v)", {"v"})

    def test_no_args_raises(self) -> None:
        """A call without arguments is refused."""
        with pytest.raises(ValueError, match="requires at least 1 argument"):
            _emit("exp()")

    def test_non_name_call_raises(self) -> None:
        """An attribute call such as obj.method(v) is refused."""
        with pytest.raises(ValueError, match="Only named function calls"):
            _emit("obj.method(v)", {"v"})
