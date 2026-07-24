# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStuckAtFaults from former test_fault_tolerance.py

"""Focused suite: TestStuckAtFaults from former test_fault_tolerance.py."""

from __future__ import annotations

from tests.fault_tolerance_support import *  # noqa: F403


class TestStuckAtFaults:
    """Test stuck-at fault models."""

    def test_stuck_at_0_decreases_probability(self):
        """Stuck-at-0 faults should decrease the estimated probability."""
        np.random.seed(42)
        p_true = 0.7
        bs = generate_bernoulli_bitstream(p_true, 5000)
        corrupted = FaultInjector.inject_stuck_at(bs, 0.2, value=0)
        p_est = bitstream_to_probability(corrupted)
        assert p_est < p_true

    def test_stuck_at_1_increases_probability(self):
        """Stuck-at-1 faults should increase the estimated probability."""
        np.random.seed(42)
        p_true = 0.3
        bs = generate_bernoulli_bitstream(p_true, 5000)
        corrupted = FaultInjector.inject_stuck_at(bs, 0.2, value=1)
        p_est = bitstream_to_probability(corrupted)
        assert p_est > p_true
