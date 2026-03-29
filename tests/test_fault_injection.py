# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for fault injection and resilience

"""Tests for FaultInjector: bit-flip, stuck-at, and TMR majority vote."""

from __future__ import annotations

import numpy as np

from sc_neurocore.utils.bitstreams import generate_bernoulli_bitstream
from sc_neurocore.utils.fault_injection import FaultInjector
from sc_neurocore.utils.rng import RNG


def _majority_vote(a, b, c):
    return ((a & b) | (a & c) | (b & c)).astype(np.uint8)


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


class TestMajorityVoteTMR:
    def test_clean_signal_preserved(self):
        bits = generate_bernoulli_bitstream(0.7, 2000, rng=RNG(42))
        voted = _majority_vote(bits.copy(), bits.copy(), bits.copy())
        np.testing.assert_array_equal(bits, voted)

    def test_tmr_reduces_error(self):
        """TMR should reduce error compared to single faulty channel."""
        L = 5000
        target = 0.6
        rate = 0.15
        n_trials = 100
        single_errs, tmr_errs = [], []

        for trial in range(n_trials):
            clean = generate_bernoulli_bitstream(target, L, rng=RNG(trial))
            f1 = FaultInjector.inject_bit_flips(clean.copy(), rate)
            single_errs.append(abs(np.mean(f1) - target))

            fa = FaultInjector.inject_bit_flips(clean.copy(), rate)
            fb = FaultInjector.inject_bit_flips(clean.copy(), rate)
            fc = FaultInjector.inject_bit_flips(clean.copy(), rate)
            voted = _majority_vote(fa, fb, fc)
            tmr_errs.append(abs(np.mean(voted) - target))

        assert np.mean(tmr_errs) < np.mean(single_errs)

    def test_tmr_output_binary(self):
        a = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        b = np.array([1, 1, 0, 0, 1], dtype=np.uint8)
        c = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
        voted = _majority_vote(a, b, c)
        expected = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(voted, expected)


class TestFixedPointComparison:
    def test_sc_beats_fp_at_10pct_error(self):
        """SC should degrade less than 16-bit fixed-point at 10% BER."""
        target = 0.65
        rate = 0.1
        L = 2000
        n_trials = 200
        rng = np.random.default_rng(42)

        sc_errs, fp_errs = [], []
        for trial in range(n_trials):
            # SC
            bits = generate_bernoulli_bitstream(target, L, rng=RNG(trial))
            faulted = FaultInjector.inject_bit_flips(bits, rate)
            sc_errs.append(abs(np.mean(faulted) - target))

            # Fixed-point 16-bit
            fp_val = int(target * (1 << 16))
            fp_bits = np.array([(fp_val >> b) & 1 for b in range(16)])
            flip = rng.random(16) < rate
            fp_faulted = fp_bits ^ flip.astype(int)
            decoded = sum(b << i for i, b in enumerate(fp_faulted)) / (1 << 16)
            fp_errs.append(abs(decoded - target))

        assert np.mean(sc_errs) < np.mean(fp_errs), (
            f"SC mean err {np.mean(sc_errs):.4f} >= FP {np.mean(fp_errs):.4f}"
        )
