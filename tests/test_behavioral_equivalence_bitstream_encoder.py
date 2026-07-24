# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamEncoder from former test_behavioral_equivalence.py

"""Focused suite: TestBitstreamEncoder from former test_behavioral_equivalence.py."""

from __future__ import annotations

from tests.behavioral_equivalence_support import *  # noqa: F403


class TestBitstreamEncoder:
    """Verify decorrelation between parallel encoder instances."""

    def test_same_seed_same_output(self):
        enc_a = FixedPointBitstreamEncoder(seed_init=0xACE1)
        enc_b = FixedPointBitstreamEncoder(seed_init=0xACE1)

        bits_a = [enc_a.step(32768) for _ in range(256)]
        bits_b = [enc_b.step(32768) for _ in range(256)]

        assert bits_a == bits_b

    def test_different_seeds_decorrelate(self):
        """Encoders with different SEED_INIT must produce different bitstreams."""
        # These match the HDL fix: input encoders use 0xACE1 + i*7
        enc_0 = FixedPointBitstreamEncoder(seed_init=0xACE1 + 0 * 7)
        enc_1 = FixedPointBitstreamEncoder(seed_init=0xACE1 + 1 * 7)

        x_val = 32768  # ~50% probability
        bits_0 = [enc_0.step(x_val) for _ in range(256)]
        bits_1 = [enc_1.step(x_val) for _ in range(256)]

        assert bits_0 != bits_1, "Same x_value with different seeds should decorrelate"

    def test_weight_encoder_decorrelation(self):
        """Weight encoders (0xBEEF base) must differ from input encoders (0xACE1 base)."""
        enc_input = FixedPointBitstreamEncoder(seed_init=0xACE1)
        enc_weight = FixedPointBitstreamEncoder(seed_init=0xBEEF)

        x_val = 32768
        bits_in = [enc_input.step(x_val) for _ in range(256)]
        bits_wt = [enc_weight.step(x_val) for _ in range(256)]

        assert bits_in != bits_wt

    def test_probability_convergence(self):
        """Over many samples, proportion of 1s should converge to x_value/65535."""
        enc = FixedPointBitstreamEncoder(seed_init=0xACE1)
        x_val = 32768  # ~0.5
        length = 10000
        ones = sum(enc.step(x_val) for _ in range(length))
        p_hat = ones / length
        assert abs(p_hat - 0.5) < 0.05, f"Expected ~0.5, got {p_hat:.3f}"

    def test_zero_input_no_ones(self):
        enc = FixedPointBitstreamEncoder(seed_init=0xACE1)
        bits = [enc.step(0) for _ in range(1000)]
        assert sum(bits) == 0

    def test_max_input_all_ones(self):
        """x_value = 2^16 - 1 should produce all 1s (LFSR < 65535 always true)."""
        enc = FixedPointBitstreamEncoder(seed_init=0xACE1)
        bits = [enc.step(65535) for _ in range(1000)]
        # Allow 1 miss because LFSR can equal 65534 which is < 65535
        assert sum(bits) >= 999
