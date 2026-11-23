# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGPUPopcount from former test_gpu_backend.py

"""Focused suite: TestGPUPopcount from former test_gpu_backend.py."""

from __future__ import annotations

from sc_neurocore.accel.gpu_backend import (
    gpu_popcount,
    to_host,
    xp,
)


class TestGPUPopcount:
    """Counting set bits in words."""

    def test_known_values(self) -> None:
        """Known words give their bit counts."""
        # 0xFF = 8 bits set, packed in uint64
        packed = xp.array([0xFF, 0xFFFF, 0], dtype=xp.uint64)
        counts = to_host(gpu_popcount(packed))
        assert counts[0] == 8
        assert counts[1] == 16
        assert counts[2] == 0

    def test_all_ones(self) -> None:
        """A full word counts 64."""
        packed = xp.array([0xFFFFFFFFFFFFFFFF], dtype=xp.uint64)
        assert int(to_host(gpu_popcount(packed))[0]) == 64

    def test_single_bit(self) -> None:
        """A single set bit counts 1."""
        for bit_pos in [0, 1, 31, 63]:
            packed = xp.array([1 << bit_pos], dtype=xp.uint64)
            assert int(to_host(gpu_popcount(packed))[0]) == 1
