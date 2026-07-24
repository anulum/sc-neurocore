# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBipolarEncodeDecode from former test_bipolar_sc.py

"""Focused suite: TestBipolarEncodeDecode from former test_bipolar_sc.py."""

from __future__ import annotations

from tests.bipolar_sc_support import *  # noqa: F403


class TestBipolarEncodeDecode:
    def test_encode_plus_one(self):
        bits = bipolar_encode(1.0, 10000, rng=np.random.default_rng(42))
        assert bits.mean() > 0.95

    def test_encode_minus_one(self):
        bits = bipolar_encode(-1.0, 10000, rng=np.random.default_rng(42))
        assert bits.mean() < 0.05

    def test_encode_zero(self):
        bits = bipolar_encode(0.0, 10000, rng=np.random.default_rng(42))
        assert 0.45 < bits.mean() < 0.55

    def test_decode_roundtrip(self):
        for v in [-0.8, -0.3, 0.0, 0.5, 0.9]:
            bits = bipolar_encode(v, 100000, rng=np.random.default_rng(42))
            decoded = bipolar_decode(bits)
            assert abs(decoded - v) < 0.02, f"v={v}, decoded={decoded}"

    def test_rejects_out_of_range_encode_values(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            bipolar_encode(2.0, 100, rng=np.random.default_rng(42))
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            bipolar_encode(-2.0, 100, rng=np.random.default_rng(42))

    def test_rejects_nonpositive_bitstream_length(self):
        with pytest.raises(ValueError, match="positive"):
            bipolar_encode(0.0, 0, rng=np.random.default_rng(42))

    def test_decode_rejects_empty_bitstream(self):
        with pytest.raises(ValueError, match="non-empty"):
            bipolar_decode(np.array([], dtype=np.uint8))
