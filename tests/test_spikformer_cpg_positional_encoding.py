# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCPGPositionalEncoding from former test_spikformer.py

"""Focused suite: TestCPGPositionalEncoding from former test_spikformer.py."""

from __future__ import annotations

from tests.spikformer_support import *  # noqa: F403


class TestCPGPositionalEncoding:
    def test_encode_shape(self):
        cpe = CPGPositionalEncoding(d_model=16)
        enc = cpe.encode(seq_len=10)
        assert enc.shape == (10, 16)

    def test_encode_range(self):
        cpe = CPGPositionalEncoding(d_model=8)
        enc = cpe.encode(seq_len=50)
        assert enc.min() >= 0.0
        assert enc.max() <= 1.0

    def test_different_positions_different_encodings(self):
        cpe = CPGPositionalEncoding(d_model=8)
        enc = cpe.encode(seq_len=10)
        assert not np.allclose(enc[0], enc[5])

    def test_encode_spikes(self):
        cpe = CPGPositionalEncoding(d_model=16)
        spikes = cpe.encode_spikes(seq_len=20)
        assert spikes.shape == (20, 16)
        assert set(np.unique(spikes)).issubset({0, 1})

    def test_encode_spikes_with_rng(self):
        cpe = CPGPositionalEncoding(d_model=8)
        rng = np.random.RandomState(42)
        s1 = cpe.encode_spikes(10, rng=np.random.RandomState(42))
        s2 = cpe.encode_spikes(10, rng=np.random.RandomState(42))
        np.testing.assert_array_equal(s1, s2)

    def test_max_len(self):
        cpe = CPGPositionalEncoding(d_model=4, max_len=100)
        assert cpe.max_len == 100
        enc = cpe.encode(seq_len=100)
        assert enc.shape == (100, 4)
