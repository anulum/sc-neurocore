# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStuckAtInjection from former test_fault_injection.py

"""Focused suite: TestStuckAtInjection from former test_fault_injection.py."""

from __future__ import annotations

from tests.fault_injection_support import *  # noqa: F403

class TestStuckAtInjection:
    def test_stuck_at_zero_decreases_probability(self):
        L = 5000
        target = 0.7
        bits = generate_bernoulli_bitstream(target, L, rng=RNG(42))
        faulted = FaultInjector.inject_stuck_at(bits, 0.2, value=0)
        assert np.mean(faulted) < target

    def test_stuck_at_one_increases_probability(self):
        L = 5000
        target = 0.3
        bits = generate_bernoulli_bitstream(target, L, rng=RNG(42))
        faulted = FaultInjector.inject_stuck_at(bits, 0.2, value=1)
        assert np.mean(faulted) > target

    def test_analytical_bound_sa0(self):
        """Stuck-at-0 bias bounded by f * p."""
        L = 10000
        p = 0.8
        f = 0.15
        bits = generate_bernoulli_bitstream(p, L, rng=RNG(42))
        faulted = FaultInjector.inject_stuck_at(bits, f, value=0)
        error = abs(np.mean(faulted) - p)
        bound = f * p + 0.02  # analytical + tolerance
        assert error < bound, f"SA0 error {error:.3f} > bound {bound:.3f}"

    def test_analytical_bound_sa1(self):
        """Stuck-at-1 bias bounded by f * (1-p)."""
        L = 10000
        p = 0.3
        f = 0.2
        bits = generate_bernoulli_bitstream(p, L, rng=RNG(42))
        faulted = FaultInjector.inject_stuck_at(bits, f, value=1)
        error = abs(np.mean(faulted) - p)
        bound = f * (1.0 - p) + 0.02
        assert error < bound, f"SA1 error {error:.3f} > bound {bound:.3f}"

    def test_zero_rate_no_change(self):
        bits = generate_bernoulli_bitstream(0.5, 500, rng=RNG(1))
        faulted = FaultInjector.inject_stuck_at(bits.copy(), 0.0, value=1)
        np.testing.assert_array_equal(bits, faulted)
