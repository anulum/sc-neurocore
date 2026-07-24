# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLFSRReplay from former test_explainability.py

"""Focused suite: TestLFSRReplay from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403


class TestLFSRReplay:
    def test_deterministic_output(self):
        a = LFSRReplay(0xACE1)
        b = LFSRReplay(0xACE1)
        for _ in range(100):
            assert a.step() == b.step()

    def test_zero_seed_raises(self):
        with pytest.raises(ValueError):
            LFSRReplay(0)

    def test_encode_length(self):
        lfsr = LFSRReplay(0xACE1)
        bs = lfsr.encode(32768, 1000)
        assert len(bs) == 1000

    def test_encode_probability(self):
        lfsr = LFSRReplay(0xACE1)
        bs = lfsr.encode(32768, 10000)
        p = np.mean(bs)
        assert abs(p - 0.5) < 0.03

    def test_reset_replays_same(self):
        lfsr = LFSRReplay(0xACE1)
        bs1 = lfsr.encode(32768, 100)
        lfsr.reset()
        bs2 = lfsr.encode(32768, 100)
        np.testing.assert_array_equal(bs1, bs2)

    def test_different_seeds_different_output(self):
        a = LFSRReplay(0xACE1)
        b = LFSRReplay(0xBEEF)
        bs_a = a.encode(32768, 100)
        bs_b = b.encode(32768, 100)
        assert not np.array_equal(bs_a, bs_b)

    def test_matches_core_engine_polynomial(self):
        lfsr = LFSRReplay(0xACE1)
        vals = [lfsr.step() for _ in range(10)]
        lfsr2 = LFSRReplay(0xACE1)
        vals2 = [lfsr2.step() for _ in range(10)]
        assert vals == vals2
