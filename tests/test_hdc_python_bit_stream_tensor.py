# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitStreamTensor from former test_hdc_python.py

"""Focused suite: TestBitStreamTensor from former test_hdc_python.py."""

from __future__ import annotations

from tests.hdc_python_support import *  # noqa: F403

class TestBitStreamTensor:
    def test_create_random(self):
        t = BitStreamTensor(1000, seed=42)
        assert len(t) == 1000
        pc = t.popcount()
        assert 350 < pc < 650, f"Expected ~500, got {pc}"

    def test_from_packed_roundtrip(self):
        t1 = BitStreamTensor(128, seed=99)
        t2 = BitStreamTensor.from_packed(t1.data, t1.length)
        assert t1.hamming_distance(t2) == 0.0

    def test_xor_self_is_zero(self):
        t = BitStreamTensor(256, seed=1)
        result = t.xor(t)
        assert result.popcount() == 0

    def test_xor_inplace(self):
        a = BitStreamTensor(256, seed=1)
        b = BitStreamTensor(256, seed=2)
        expected = a.xor(b)
        a.xor_inplace(b)
        assert a.hamming_distance(expected) == 0.0

    def test_hamming_identical_zero(self):
        t = BitStreamTensor(1000, seed=7)
        assert t.hamming_distance(t) == 0.0

    def test_hamming_random_near_half(self):
        a = BitStreamTensor(10_000, seed=42)
        b = BitStreamTensor(10_000, seed=99)
        hd = a.hamming_distance(b)
        assert 0.4 < hd < 0.6, f"Expected ~0.5, got {hd}"

    def test_rotate_identity(self):
        t = BitStreamTensor(128, seed=5)
        original_data = list(t.data)
        t.rotate_right(0)
        assert list(t.data) == original_data
        t.rotate_right(128)  # full length
        assert list(t.data) == original_data

    def test_bundle_majority(self):
        a = BitStreamTensor(10_000, seed=1)
        b = BitStreamTensor(10_000, seed=2)
        c = BitStreamTensor(10_000, seed=3)
        result = BitStreamTensor.bundle([a, b, c])
        assert len(result) == 10_000
        # Bundle of 3 random vectors should still be ~50% ones
        ratio = result.popcount() / 10_000
        assert 0.3 < ratio < 0.7

    def test_repr(self):
        t = BitStreamTensor(100, seed=42)
        r = repr(t)
        assert "BitStreamTensor" in r
        assert "100" in r
