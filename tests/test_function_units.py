# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — function lowering v2: the integer reference against the Python model

"""Tests of :mod:`sc_neurocore.compiler.function_units`.

Every unit is evaluated at every input code of Q8.8 and Q4.12 and compared with
the Python model's function clipped to the word. The stated bounds are the
largest errors measured on 2026-10-19, rounded up; a regression above them
fails. Q16.16 is sampled. The helpers are checked against exact integer facts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.compiler import function_units as fu
from sc_neurocore.neurons.equation_namespace import build_eval_namespace

_NS = build_eval_namespace()
_REFERENCE = {
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "sigmoid": _NS["sigmoid"],
    "tanh": np.tanh,
    "cosh": np.cosh,
    "sinh": np.sinh,
    "exprel": _NS["exprel"],
    "cbrt": np.cbrt,
    "exp2": np.exp2,
    "log2": np.log2,
    "log10": np.log10,
    "expm1": np.expm1,
    "log1p": np.log1p,
    "softplus": _NS["softplus"],
    "tan": np.tan,
    "arctan": np.arctan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "erf": _NS["erf"],
    "floor": np.floor,
    "ceil": np.ceil,
    "trunc": np.trunc,
    "sign": np.sign,
}
# Inputs the Python model evaluates to a real number (outside, the units state a value).
_DOMAIN = {
    "log": lambda x: x > 0,
    "log2": lambda x: x > 0,
    "log10": lambda x: x > 0,
    "log1p": lambda x: x > -1,
    "sqrt": lambda x: x >= 0,
    "cbrt": lambda x: x >= 0,
    "arcsin": lambda x: -1 <= x <= 1,
    "arccos": lambda x: -1 <= x <= 1,
}
# Largest error in least significant bits, 16-bit formats, every code (measured, rounded up).
_BOUND_16 = {name: 0.51 for name in fu.UNARY_FUNCTIONS} | {"cosh": 0.63, "sinh": 0.63, "tan": 0.52}


def _max_error_lsb(
    function: str, data_width: int, fraction: int, codes: range | list[int]
) -> float:
    scale = float(1 << fraction)
    low, high = -(1 << (data_width - 1)) / scale, ((1 << (data_width - 1)) - 1) / scale
    worst = 0.0
    for code in codes:
        x = code / scale
        if function in _DOMAIN and not _DOMAIN[function](x):
            continue
        with np.errstate(all="ignore"):
            reference = float(np.clip(_REFERENCE[function](np.float64(x)), low, high))
        worst = max(
            worst, abs(fu.evaluate(function, code, data_width, fraction) / scale - reference)
        )
    return worst * scale


class TestAccuracy:
    """Measured error against the Python model stays within the stated bounds."""

    @pytest.mark.parametrize(("dw", "fr"), [(16, 8), (16, 12)], ids=["Q8.8", "Q4.12"])
    @pytest.mark.parametrize("function", fu.UNARY_FUNCTIONS)
    def test_every_code_of_16_bit_formats(self, function: str, dw: int, fr: int) -> None:
        """Every input code is within the stated bound (correct rounding is 0.5 LSB)."""
        codes = range(-(1 << (dw - 1)), 1 << (dw - 1))
        assert _max_error_lsb(function, dw, fr, codes) <= _BOUND_16[function]

    @pytest.mark.parametrize("function", fu.UNARY_FUNCTIONS)
    def test_sampled_q16_16(self, function: str) -> None:
        """At Q16.16 bounded functions stay within 0.6 LSB; exp-like ones within 2**-28 relative."""
        codes = list(range(-(1 << 31), 1 << 31, 1 << 20)) + list(range(-(1 << 20), 1 << 20, 4099))
        error = _max_error_lsb(function, 32, 16, codes)
        if function in ("exp", "cosh", "sinh", "exprel", "exp2", "expm1"):
            assert error <= 5.0  # at most ~2**-28.6 of the largest value, 32767.99
        elif function == "tan":
            pass  # ill-conditioned at the poles; see test_tan_away_from_the_poles
        else:
            assert error <= 0.6

    def test_tan_away_from_the_poles(self) -> None:
        """At Q16.16, where |tan x| <= 10 the unit is within 0.6 LSB of the model.

        Near a pole the error grows with the conditioning, about
        ``(1 + tan²x)·2π·2**-30`` from the turn's 30-bit resolution: 2.4 LSB at
        ``|tan x| <= 100`` (measured 2026-10-19).
        """
        codes = [c for c in range(-(1 << 22), 1 << 22, 997) if abs(np.tan(c / 65536.0)) <= 10.0]
        assert _max_error_lsb("tan", 32, 16, codes) <= 0.6


class TestHelpers:
    """Integer helpers equal the arithmetic they name."""

    @pytest.mark.parametrize("value", [0, 1, 2, 3, 15, 16, 17, 2**40 + 5, 2**62 + 2**31])
    def test_isqrt_is_the_floor_square_root(self, value: int) -> None:
        """``isqrt`` equals ``math.isqrt`` up to ``2**63``."""
        assert fu.isqrt(value) == math.isqrt(value)

    def test_shift_round_rounds_half_up(self) -> None:
        """Right shifts round half up and left shifts are exact."""
        assert fu.shift_round(5, -1) == 3 and fu.shift_round(-5, -1) == -2
        assert fu.shift_round(3, 4) == 48

    def test_trunc_div_truncates_towards_zero(self) -> None:
        """Division truncates as C and Verilog do."""
        assert fu.trunc_div(-7, 2) == -3 and fu.trunc_div(7, -2) == -3 and fu.trunc_div(7, 2) == 3

    def test_scale_to_code_saturates_before_forming_a_wide_value(self) -> None:
        """A left shift that leaves the word saturates instead."""
        assert fu.scale_to_code(1 << 20, 20, 16) == (1 << 15) - 1
        assert fu.scale_to_code(-(1 << 20), 20, 16) == -(1 << 15)
        assert fu.scale_to_code(3, 2, 16) == 12

    def test_tables_interpolate_their_end_points(self) -> None:
        """At segment boundaries the quadratic equals the tabulated function."""
        assert fu.interpolate("exp2", 0, fu.B) == 1 << fu.P
        assert fu.interpolate("exp2", 1 << fu.B, fu.B) == round(2.0 * (1 << fu.P))
        assert abs(fu.interpolate("qsin", 1 << 28, 28) - (1 << fu.P)) <= 1


class TestDomainStatements:
    """Inputs outside the Python model's domain have their stated values."""

    def test_log_of_non_positive_is_the_most_negative_code(self) -> None:
        """``log`` of zero or a negative input is the most negative code."""
        assert fu.evaluate("log", 0, 16, 8) == -(1 << 15)
        assert fu.evaluate("log", -5, 16, 8) == -(1 << 15)

    def test_sqrt_of_negative_is_zero_and_cbrt_is_odd(self) -> None:
        """``sqrt`` of a negative input is 0; ``x**(1/3)`` is odd in ``x``."""
        assert fu.evaluate("sqrt", -256, 16, 8) == 0
        assert fu.evaluate("cbrt", -2048, 16, 8) == -fu.evaluate("cbrt", 2048, 16, 8)

    def test_inverse_sine_and_cosine_saturate_outside_the_unit_interval(self) -> None:
        """``arcsin`` and ``arccos`` of ``|x| > 1`` are their values at ``±1``."""
        assert fu.evaluate("arcsin", 3 << 8, 16, 8) == fu.evaluate("arcsin", 1 << 8, 16, 8)
        assert fu.evaluate("arccos", -(3 << 8), 16, 8) == fu.evaluate("arccos", -(1 << 8), 16, 8)

    def test_exact_functions_are_exact(self) -> None:
        """floor, ceil, trunc and sign at a few codes, including negatives."""
        assert [fu.evaluate("floor", c, 16, 8) for c in (-257, -256, 255)] == [-512, -256, 0]
        assert [fu.evaluate("ceil", c, 16, 8) for c in (-257, 1, 256)] == [-256, 256, 256]
        assert [fu.evaluate("trunc", c, 16, 8) for c in (-257, 257)] == [-256, 256]
        assert [fu.evaluate("sign", c, 8, 7) for c in (-5, 0, 5)] == [-128, 0, 127]

    def test_exprel_at_zero_is_one(self) -> None:
        """The removable singularity: ``exprel(0) = 1`` at every format."""
        for dw, fr in [(8, 2), (16, 8), (32, 24)]:
            assert fu.evaluate("exprel", 0, dw, fr) == 1 << fr


class TestFormats:
    """Formats the 64-bit units cannot mirror are refused."""

    @pytest.mark.parametrize(("dw", "fr"), [(33, 16), (1, 0), (16, 16), (32, 30)])
    def test_refused(self, dw: int, fr: int) -> None:
        """Words above 32 bits, fractions not below the width, and ``dw + fr > 61``."""
        with pytest.raises(ValueError):
            fu.evaluate("exp", 0, dw, fr)

    def test_unknown_function_is_refused(self) -> None:
        """A name outside the unit registry has no v2 unit."""
        with pytest.raises(ValueError, match="no v2 unit"):
            fu.evaluate("gamma", 0, 16, 8)
