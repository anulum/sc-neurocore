# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGPUPack from former test_gpu_backend.py

"""Focused suite: TestGPUPack from former test_gpu_backend.py."""

from __future__ import annotations

from tests.gpu_backend_support import *  # noqa: F403

class TestGPUPack:
    def test_1d_roundtrip(self):
        bits = np.array([1, 0, 1, 1] + [0] * 60, dtype=np.uint8)
        packed = gpu_pack_bitstream(to_device(bits))
        host = to_host(packed)
        assert host.dtype == np.uint64
        assert host.shape == (1,)
        # bit0=1, bit1=0, bit2=1, bit3=1 -> 0b1101 = 13
        assert int(host[0]) == 13

    def test_2d_shape(self):
        bits = xp.zeros((4, 128), dtype=xp.uint8)
        packed = gpu_pack_bitstream(bits)
        assert packed.shape == (4, 2)  # 128/64 = 2 words

    def test_all_ones(self):
        bits = xp.ones(64, dtype=xp.uint8)
        packed = gpu_pack_bitstream(bits)
        assert int(to_host(packed)[0]) == (2**64 - 1)

    def test_all_zeros(self):
        bits = xp.zeros(64, dtype=xp.uint8)
        packed = gpu_pack_bitstream(bits)
        assert int(to_host(packed)[0]) == 0
