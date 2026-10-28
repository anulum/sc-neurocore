# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — C++ power lowering

"""Focused suite: TestPow from former test_c_expr_emitter.py."""

from __future__ import annotations

from tests.c_expr_emitter_support import *  # noqa: F403


class TestPow:
    """Check how the C++ emitter lowers powers to products and roots."""

    def test_square_is_repeated_multiply(self) -> None:
        """Check that a square lowers to one multiplication."""
        assert _emit("v ** 2", {"v"}) == "(v * v)"

    def test_cube(self) -> None:
        """Check that a cube lowers to two multiplications."""
        assert _emit("v ** 3", {"v"}) == "(v * v * v)"

    def test_eighth_power(self) -> None:
        """Check that an eighth power lowers to seven multiplications."""
        assert _emit("v ** 8", {"v"}).count("v *") == 7

    def test_sqrt_from_half_power(self) -> None:
        """Check that a half power lowers to ``hls::sqrt``."""
        assert _emit("v ** 0.5", {"v"}) == "hls::sqrt(v)"

    def test_cbrt_from_third_power(self) -> None:
        """Check that a one-third power lowers to ``hls::cbrt``."""
        assert _emit("v ** (1.0 / 3.0)", {"v"}) == "hls::cbrt(v)"

    def test_general_constant_uses_math_power(self) -> None:
        """Check that a general constant reaches the real library power."""
        assert _emit("v ** 9", {"v"}) == "hls::pow(v, fp_t(9.0))"
