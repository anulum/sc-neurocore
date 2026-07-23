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
    def test_direct_math_functions(self) -> None:
        assert _emit("exp(v)", {"v"}) == "hls::exp(v)"
        assert _emit("tanh(v)", {"v"}) == "hls::tanh(v)"
        assert _emit("cosh(v)", {"v"}) == "hls::cosh(v)"
        assert _emit("sin(v)", {"v"}) == "hls::sin(v)"
        assert _emit("log(v)", {"v"}) == "hls::log(v)"

    def test_std_namespace_option(self) -> None:
        assert _emit("exp(v)", {"v"}, math_ns="std") == "std::exp(v)"

    def test_sigmoid_and_expit_use_helper(self) -> None:
        assert _emit("sigmoid(v)", {"v"}) == "sc_sigmoid(v)"
        assert _emit("expit(v)", {"v"}) == "sc_sigmoid(v)"

    def test_exprel_uses_helper(self) -> None:
        assert _emit("exprel(v)", {"v"}) == "sc_exprel(v)"

    def test_abs(self) -> None:
        assert _emit("abs(v)", {"v"}) == "hls::abs(v)"

    def test_clip_three_args(self) -> None:
        out = _emit("clip(v, 0.0, 1.0)", {"v"})
        assert "? fp_t(0.0)" in out and "? fp_t(1.0)" in out

    def test_max_min(self) -> None:
        assert _emit("max(a, b)") == "((a > b) ? a : b)"
        assert _emit("min(a, b)") == "((a < b) ? a : b)"

    def test_clip_single_arg_passthrough(self) -> None:
        # clip with only the value (no bounds) passes the argument through.
        assert _emit("clip(v)", {"v"}) == "v"

    def test_max_single_arg_passthrough(self) -> None:
        assert _emit("max(v)", {"v"}) == "v"

    def test_unsupported_function_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported function"):
            _emit("gamma(v)", {"v"})

    def test_no_args_raises(self) -> None:
        with pytest.raises(ValueError, match="requires at least 1 argument"):
            _emit("exp()")

    def test_non_name_call_raises(self) -> None:
        with pytest.raises(ValueError, match="Only named function calls"):
            _emit("obj.method(v)", {"v"})
