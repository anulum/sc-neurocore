# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLayerErrorBudget from former test_sc_scope.py

"""Focused suite: TestLayerErrorBudget from former test_sc_scope.py."""

from __future__ import annotations

from sc_neurocore.debug.sc_scope import (
    LayerErrorBudget,
)


class TestLayerErrorBudget:
    """A layer's error budget on density."""

    def test_within_tolerance(self) -> None:
        """A density inside the tolerance passes."""
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.1)
        assert eb.check(0.52) is True

    def test_outside_tolerance(self) -> None:
        """A density outside the tolerance fails."""
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.01)
        assert eb.check(0.7) is False

    def test_violations(self) -> None:
        """A failed check is counted as a violation."""
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.05)
        eb.check(0.5)  # OK
        eb.check(0.8)  # violation
        eb.check(0.5)  # OK
        assert eb.violations == 1

    def test_pass_rate(self) -> None:
        """Nine passes in ten give a pass rate of 0.9."""
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.05)
        for _ in range(9):
            eb.check(0.5)
        eb.check(0.9)  # 1 violation
        assert abs(eb.pass_rate - 0.9) < 0.01

    def test_mean_error(self) -> None:
        """The mean error averages the checked errors."""
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.1)
        eb.check(0.6)  # err=0.1
        eb.check(0.4)  # err=0.1
        assert abs(eb.mean_error - 0.1) < 0.01

    def test_max_error(self) -> None:
        """The largest checked error is reported."""
        eb = LayerErrorBudget(0, expected_density=0.5, tolerance=0.1)
        eb.check(0.5)
        eb.check(0.9)
        assert abs(eb.max_error - 0.4) < 0.01
