# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitFlipDegradation from former test_fault_tolerance.py

"""Focused suite: TestBitFlipDegradation from former test_fault_tolerance.py."""

from __future__ import annotations

from tests.fault_tolerance_support import *  # noqa: F403


class TestBitFlipDegradation:
    """Sweep bit-flip rates and verify graceful degradation."""

    @pytest.mark.parametrize("error_rate", [0.01, 0.05, 0.10, 0.25, 0.50])
    def test_sc_graceful_degradation(self, error_rate):
        """SC bitstream error should scale linearly with bit-flip rate."""
        np.random.seed(42)
        p_true = 0.7
        length = 10000
        n_trials = 20
        errors = []

        for _ in range(n_trials):
            bs = generate_bernoulli_bitstream(p_true, length)
            corrupted = FaultInjector.inject_bit_flips(bs, error_rate)
            p_est = bitstream_to_probability(corrupted)
            errors.append(abs(p_est - p_true))

        mean_error = np.mean(errors)
        # SC error is bounded: max error from bit-flips ≈ error_rate
        # (flipping a fraction f of bits shifts probability by at most f)
        assert mean_error < error_rate + 0.05

    def test_sc_vs_fixedpoint_robustness(self):
        """SC bitstream is more robust to random bit-flips than fixed-point."""
        np.random.seed(42)
        error_rate = 0.1
        p_true = 0.7
        length = 1024
        n_trials = 50

        # SC: flip 10% of bitstream bits
        sc_errors = []
        for _ in range(n_trials):
            bs = generate_bernoulli_bitstream(p_true, length)
            corrupted = FaultInjector.inject_bit_flips(bs, error_rate)
            sc_errors.append(abs(bitstream_to_probability(corrupted) - p_true))

        # Fixed-point Q8.8: flip 10% of the 16 bits
        fp_errors = []
        q_val = int(round(p_true * 256))  # Q8.8 representation
        for _ in range(n_trials):
            bits = q_val
            for bit_pos in range(16):
                if np.random.random() < error_rate:
                    bits ^= 1 << bit_pos
            # Interpret as signed Q8.8
            if bits >= 32768:
                bits -= 65536
            fp_est = bits / 256.0
            fp_errors.append(abs(fp_est - p_true))

        # SC should have significantly smaller mean error
        assert np.mean(sc_errors) < np.mean(fp_errors)

    @pytest.mark.parametrize("error_rate", [0.01, 0.10, 0.25])
    def test_longer_bitstream_more_robust(self, error_rate):
        """Longer bitstreams should be more robust to bit-flips."""
        np.random.seed(42)
        p_true = 0.6
        n_trials = 30

        errors_short = []
        errors_long = []
        for _ in range(n_trials):
            bs_short = generate_bernoulli_bitstream(p_true, 256)
            bs_long = generate_bernoulli_bitstream(p_true, 4096)
            c_short = FaultInjector.inject_bit_flips(bs_short, error_rate)
            c_long = FaultInjector.inject_bit_flips(bs_long, error_rate)
            errors_short.append(abs(bitstream_to_probability(c_short) - p_true))
            errors_long.append(abs(bitstream_to_probability(c_long) - p_true))

        # Longer bitstream → more averaging → smaller variance
        assert np.std(errors_long) <= np.std(errors_short) + 0.01
