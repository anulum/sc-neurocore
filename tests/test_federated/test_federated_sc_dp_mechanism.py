# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDPMechanism from former test_federated_sc.py

"""Focused suite: TestDPMechanism from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403

class TestDPMechanism:
    def test_flip_probability_range(self):
        dp = DPMechanism(epsilon=1.0)
        p = dp.flip_probability
        assert 0.0 < p < 0.5

    def test_higher_epsilon_less_noise(self):
        dp_low = DPMechanism(epsilon=0.5)
        dp_high = DPMechanism(epsilon=5.0)
        assert dp_high.flip_probability < dp_low.flip_probability

    def test_privatise_preserves_length(self):
        dp = DPMechanism(epsilon=1.0)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.5, 0xACE1, 256)
        noisy = dp.privatise(bs, rng)
        assert len(noisy) == len(bs)

    def test_privatise_changes_bits(self):
        dp = DPMechanism(epsilon=0.1)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.5, 0xACE1, 1000)
        noisy = dp.privatise(bs, rng)
        diff = np.sum(bs != noisy)
        assert diff > 0

    def test_high_epsilon_preserves_most_bits(self):
        dp = DPMechanism(epsilon=10.0)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.5, 0xACE1, 1000)
        noisy = dp.privatise(bs, rng)
        diff = np.sum(bs != noisy)
        assert diff < 100

    def test_per_bit_epsilon(self):
        dp = DPMechanism(epsilon=2.0)
        assert dp.per_bit_epsilon() > 0

    def test_per_bit_epsilon_degenerate_flip_probability_is_infinite(self):
        # A deeply negative epsilon drives the flip probability to 1.0 (every
        # bit flipped), where ln((1-p)/p) is undefined: the per-bit cost is
        # reported as infinite rather than raising.
        dp = DPMechanism(epsilon=-800.0)
        assert dp.flip_probability >= 1.0
        assert dp.per_bit_epsilon() == float("inf")

    def test_total_epsilon(self):
        dp = DPMechanism(epsilon=1.0)
        total = dp.total_epsilon(256)
        assert total > 0
