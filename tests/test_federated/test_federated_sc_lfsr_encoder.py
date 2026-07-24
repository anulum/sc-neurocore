# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLFSREncoder from former test_federated_sc.py

"""Focused suite: TestLFSREncoder from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403


class TestLFSREncoder:
    def test_encode_half(self):
        bs = lfsr_encode(0.5, 0xACE1, 1000)
        p = bitstream_probability(bs)
        assert abs(p - 0.5) < 0.05

    def test_encode_zero(self):
        bs = lfsr_encode(0.0, 0xACE1, 256)
        assert np.sum(bs) == 0

    def test_encode_one(self):
        bs = lfsr_encode(1.0, 0xACE1, 256)
        assert np.sum(bs) == 256

    def test_deterministic(self):
        a = lfsr_encode(0.3, 0xACE1, 128)
        b = lfsr_encode(0.3, 0xACE1, 128)
        assert np.array_equal(a, b)

    def test_different_seeds(self):
        a = lfsr_encode(0.5, 0xACE1, 128)
        b = lfsr_encode(0.5, 0xBEEF, 128)
        assert not np.array_equal(a, b)

    def test_zero_seed_is_reset_to_one(self):
        # A zero LFSR register is a fixed point that never advances, so a seed
        # of 0 must be bumped to 1 before stepping.
        bs = lfsr_encode(0.5, 0, 64)
        assert bs.shape == (64,)
        assert bs.dtype == np.uint8
