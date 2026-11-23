# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConvergenceTracker from former test_federated_sc.py

"""Focused suite: TestConvergenceTracker from former test_federated_sc.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.federated.federated_sc import (
    ConvergenceTracker,
)


class TestConvergenceTracker:
    """Tracking convergence from gradient norms."""

    def test_not_converged_initially(self) -> None:
        """A new tracker has not converged."""
        ct = ConvergenceTracker()
        assert not ct.converged

    def test_converged_after_stable_norms(self) -> None:
        """Small stable norms count as converged."""
        ct = ConvergenceTracker()
        for _ in range(10):
            ct.record(np.array([0.001, 0.002]))
        assert ct.converged

    def test_not_converged_if_large_norm(self) -> None:
        """A large final norm undoes convergence."""
        ct = ConvergenceTracker()
        for _ in range(4):
            ct.record(np.array([0.001, 0.002]))
        ct.record(np.array([10.0, 20.0]))
        assert not ct.converged

    def test_trend_decreasing(self) -> None:
        """Falling norms give a decreasing trend."""
        ct = ConvergenceTracker()
        ct.record(np.array([10.0]))
        ct.record(np.array([5.0]))
        assert ct.trend == "decreasing"

    def test_trend_increasing(self) -> None:
        """Rising norms give an increasing trend."""
        ct = ConvergenceTracker()
        ct.record(np.array([5.0]))
        ct.record(np.array([10.0]))
        assert ct.trend == "increasing"

    def test_trend_insufficient(self) -> None:
        """Too few norms give no trend."""
        ct = ConvergenceTracker()
        assert ct.trend == "insufficient_data"

    def test_trend_stable_on_equal_norms(self) -> None:
        """Equal norms give a stable trend."""
        # Two consecutive rounds with the same gradient norm are neither rising
        # nor falling, so the trend is reported as stable.
        ct = ConvergenceTracker()
        ct.record(np.array([3.0, 4.0]))  # norm 5.0
        ct.record(np.array([4.0, 3.0]))  # norm 5.0
        assert ct.trend == "stable"

    def test_record_loss(self) -> None:
        """Recorded losses are kept per round."""
        ct = ConvergenceTracker()
        ct.record_loss(1.5)
        ct.record_loss(1.2)
        assert len(ct.round_losses) == 2
