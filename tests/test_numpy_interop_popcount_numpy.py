# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopcountNumpy from former test_numpy_interop.py

"""Focused suite: TestPopcountNumpy from former test_numpy_interop.py."""

from __future__ import annotations

from tests.numpy_interop_support import *  # noqa: F403

class TestPopcountNumpy:
    """Zero-copy popcount_numpy tests."""

    def test_basic(self):
        packed = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        assert v3.popcount_numpy(packed) == 64

    def test_known_value(self):
        packed = np.array([0x0F0F0F0F0F0F0F0F], dtype=np.uint64)
        assert v3.popcount_numpy(packed) == 32

    def test_consistency_with_pack(self):
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 10000).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        count = v3.popcount_numpy(packed)
        assert count == int(bits.sum())

    def test_large_array(self):
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 1_000_000).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        count = v3.popcount_numpy(packed)
        assert count == int(bits.sum())

    def test_empty(self):
        packed = np.array([], dtype=np.uint64)
        assert v3.popcount_numpy(packed) == 0
