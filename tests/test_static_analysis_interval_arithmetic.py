# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIntervalArithmetic from former test_static_analysis.py

"""Focused suite: TestIntervalArithmetic from former test_static_analysis.py."""

from __future__ import annotations

from tests.static_analysis_support import *  # noqa: F403


class TestIntervalArithmetic:
    """Test the Interval class used in overflow proofs."""

    def test_addition(self) -> None:
        """[1,2] + [3,4] = [4,6]."""
        r = Interval(1, 2) + Interval(3, 4)
        assert r.lo == 4 and r.hi == 6

    def test_subtraction(self) -> None:
        """[1,5] - [2,3] = [-2,3]."""
        r = Interval(1, 5) - Interval(2, 3)
        assert r.lo == -2 and r.hi == 3

    def test_multiplication(self) -> None:
        """[-2,3] * [1,4] = [-8,12]."""
        r = Interval(-2, 3) * Interval(1, 4)
        assert r.lo == -8 and r.hi == 12

    def test_division(self) -> None:
        """[6,12] / [2,3] = [2,6]."""
        r = Interval(6, 12) / Interval(2, 3)
        assert r.lo == 2.0 and r.hi == 6.0

    def test_division_by_zero(self) -> None:
        """Division by interval containing zero returns (-inf, inf)."""
        r = Interval(1, 2) / Interval(-1, 1)
        assert r.lo == float("-inf")

    def test_negation(self) -> None:
        """-[2,5] = [-5,-2]."""
        r = -Interval(2, 5)
        assert r.lo == -5 and r.hi == -2

    def test_contains(self) -> None:
        """[3,7] is contained in [-128, 127]."""
        assert Interval(3, 7).contains(-128, 127)
        assert not Interval(-200, 7).contains(-128, 127)
