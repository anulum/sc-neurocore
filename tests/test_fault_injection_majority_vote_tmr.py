# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMajorityVoteTMR from former test_fault_injection.py

"""Focused suite: TestMajorityVoteTMR from former test_fault_injection.py."""

from __future__ import annotations

from tests.fault_injection_support import *  # noqa: F403

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
