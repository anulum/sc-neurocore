# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitFlipInjection from former test_fault_injection.py

"""Focused suite: TestBitFlipInjection from former test_fault_injection.py."""

from __future__ import annotations

from tests.fault_injection_support import *  # noqa: F403


class TestBitFlipInjection:
    def test_zero_rate_no_change(self):
        bits = generate_bernoulli_bitstream(0.5, 1000, rng=RNG(42))
        faulted = FaultInjector.inject_bit_flips(bits.copy(), 0.0)
        np.testing.assert_array_equal(bits, faulted)

    def test_full_rate_inverts_all(self):
        bits = np.ones(100, dtype=np.uint8)
        faulted = FaultInjector.inject_bit_flips(bits.copy(), 1.0)
        assert np.mean(faulted) < 0.1, "100% flip should invert nearly all"

    def test_output_still_binary(self):
        bits = generate_bernoulli_bitstream(0.7, 500, rng=RNG(1))
        faulted = FaultInjector.inject_bit_flips(bits, 0.3)
        assert set(np.unique(faulted)).issubset({0, 1})

    def test_preserves_length(self):
        bits = generate_bernoulli_bitstream(0.5, 1234, rng=RNG(10))
        faulted = FaultInjector.inject_bit_flips(bits, 0.1)
        assert len(faulted) == 1234

    def test_sc_graceful_degradation(self):
        """SC error at 10% bit-flip should be ~10% of probability range."""
        L = 10000
        target = 0.6
        bits = generate_bernoulli_bitstream(target, L, rng=RNG(42))
        faulted = FaultInjector.inject_bit_flips(bits, 0.1)
        error = abs(np.mean(faulted) - target)
        assert error < 0.15, f"SC error {error:.3f} too large for 10% flip rate"

    def test_error_scales_with_rate(self):
        """Higher flip rate should produce larger error."""
        L = 5000
        target = 0.5
        bits = generate_bernoulli_bitstream(target, L, rng=RNG(42))
        e1 = abs(np.mean(FaultInjector.inject_bit_flips(bits.copy(), 0.05)) - target)
        e2 = abs(np.mean(FaultInjector.inject_bit_flips(bits.copy(), 0.3)) - target)
        assert e2 > e1 - 0.05, "higher rate should produce larger error"
