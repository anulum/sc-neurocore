# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSIMDPack from former test_engine_v3_bitstream_kernels.py

"""Focused suite: TestSIMDPack from former test_engine_v3_bitstream_kernels.py."""

from __future__ import annotations

from tests.engine_v3_bitstream_kernels_support import *  # noqa: F403


class TestSIMDPack:
    """Test SIMD-accelerated pack_bitstream_numpy correctness."""

    def test_pack_numpy_matches_list_pack(self) -> None:
        """SIMD pack must produce identical output to list pack."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 10_000).astype(np.uint8)
        packed_list = v3.pack_bitstream(bits.tolist())
        packed_numpy = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(packed_list, packed_numpy)

    @pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128, 256, 1024, 4096])
    def test_pack_numpy_various_lengths(self, length: int) -> None:
        """SIMD pack handles all lengths including non-aligned."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, length).astype(np.uint8)
        packed_list = v3.pack_bitstream(bits.tolist())
        packed_numpy = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(packed_list, packed_numpy)

    def test_pack_numpy_deterministic(self) -> None:
        """Same input -> same output."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 128, dtype=np.uint8)
        a = np.asarray(v3.pack_bitstream_numpy(bits))
        b = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(a, b)

    def test_pack_unpack_roundtrip(self) -> None:
        """Pack->unpack roundtrip preserves bits."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 2048).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        unpacked = v3.unpack_bitstream_numpy(packed, len(bits))
        np.testing.assert_array_equal(bits, np.asarray(unpacked))
