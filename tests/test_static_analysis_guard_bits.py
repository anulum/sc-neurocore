# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGuardBits from former test_static_analysis.py

"""Focused suite: TestGuardBits from former test_static_analysis.py."""

from __future__ import annotations

from tests.static_analysis_support import *  # noqa: F403

class TestGuardBits:
    """Test guard-bit auto-computation from AST analysis."""

    def test_no_additions(self) -> None:
        """Expression with only multiplication needs 0 guard bits."""
        assert compute_guard_bits("a * b") == 0

    def test_single_addition(self) -> None:
        """One addition needs 1 guard bit."""
        assert compute_guard_bits("a + b") == 1

    def test_single_subtraction(self) -> None:
        """Subtraction also counts as an addition."""
        assert compute_guard_bits("a - b") == 1

    def test_three_additions(self) -> None:
        """Three additions (4 terms) needs 2 guard bits."""
        assert compute_guard_bits("a + b + c + d") >= 2

    def test_complex_ode(self) -> None:
        """LIF ODE: -(v - v_rest) / tau_m + R * I / C has additions."""
        bits = compute_guard_bits("-(v - v_rest) / tau_m + R * I / C")
        assert bits >= 1

    def test_multi_variable(self) -> None:
        """Multi-ODE system returns per-variable guard bits."""
        eqs = {
            "v": "-(v - v_rest) / tau + I",
            "u": "a * (b * v - u)",
        }
        result = compute_guard_bits_multi(eqs)
        assert "v" in result
        assert "u" in result
        assert result["v"] >= 1
        assert result["u"] >= 1

    def test_constant_expression(self) -> None:
        """A constant has 0 additions."""
        assert compute_guard_bits("42") == 0

    def test_nested_multiply(self) -> None:
        """Nested multiplies with no additions need 0 guard bits."""
        assert compute_guard_bits("a * b * c") == 0
