# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopcountRoutesNumpyZeroCopy from former test_numpy_interop.py

"""Focused suite: TestPopcountRoutesNumpyZeroCopy from former test_numpy_interop.py."""

from __future__ import annotations

from tests.numpy_interop_support import *  # noqa: F403

class TestPopcountRoutesNumpyZeroCopy:
    """``popcount`` takes the zero-copy numpy path yet stays result-identical (KR-4)."""

    def test_numpy_matches_popcount_numpy(self):
        packed = np.array([0xFFFFFFFFFFFFFFFF, 0x0F0F0F0F0F0F0F0F], dtype=np.uint64)
        assert v3.popcount(packed) == v3.popcount_numpy(packed) == 96

    def test_numpy_matches_list_variant(self):
        """The fast numpy path and the list-extract path agree on a large bitstream."""
        rng = np.random.RandomState(7)
        bits = rng.randint(0, 2, 10000).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        assert v3.popcount(packed) == v3.popcount(packed.tolist()) == int(bits.sum())

    def test_numpy_empty(self):
        assert v3.popcount(np.array([], dtype=np.uint64)) == 0
