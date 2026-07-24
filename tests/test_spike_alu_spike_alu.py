# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeALU from former test_spike_alu.py

"""Focused suite: TestSpikeALU from former test_spike_alu.py."""

from __future__ import annotations

from tests.spike_alu_support import *  # noqa: F403


class TestSpikeALU:
    def test_add_basic(self):
        alu = SpikeALU(n_bits=8)
        result, carry = alu.add(10, 20)
        assert result == 30
        assert not carry

    def test_add_overflow(self):
        alu = SpikeALU(n_bits=8)
        result, carry = alu.add(200, 100)
        assert result == (200 + 100) & 0xFF
        assert carry

    def test_add_zero(self):
        alu = SpikeALU(n_bits=8)
        result, carry = alu.add(0, 0)
        assert result == 0
        assert not carry

    @pytest.mark.parametrize("a,b", [(50, 20), (255, 255), (0, 0), (100, 99)])
    def test_sub_matches_python(self, a, b):
        alu = SpikeALU(n_bits=8)
        result, _ = alu.sub(a, b)
        assert result == (a - b) & 0xFF

    def test_bitwise_and(self):
        alu = SpikeALU(n_bits=8)
        assert alu.bitwise_and(0xFF, 0x0F) == 0x0F

    def test_bitwise_or(self):
        alu = SpikeALU(n_bits=8)
        assert alu.bitwise_or(0xF0, 0x0F) == 0xFF

    def test_bitwise_xor(self):
        alu = SpikeALU(n_bits=8)
        assert alu.bitwise_xor(0xAA, 0x55) == 0xFF
        assert alu.bitwise_xor(42, 42) == 0

    @pytest.mark.parametrize("a,b,expected", [(10, 20, -1), (20, 10, 1), (15, 15, 0)])
    def test_compare(self, a, b, expected):
        alu = SpikeALU(n_bits=8)
        assert alu.compare(a, b) == expected

    def test_shift_left(self):
        alu = SpikeALU(n_bits=8)
        assert alu.shift_left(1) == 2
        assert alu.shift_left(0b01010101) == (0b10101010 & 0xFF)

    def test_shift_right(self):
        alu = SpikeALU(n_bits=8)
        assert alu.shift_right(2) == 1
        assert alu.shift_right(1) == 0

    def test_add_commutative(self):
        alu = SpikeALU(n_bits=8)
        for _ in range(20):
            a = np.random.randint(0, 256)
            b = np.random.randint(0, 256)
            r1, _ = alu.add(a, b)
            r2, _ = alu.add(b, a)
            assert r1 == r2, f"add not commutative: {a}+{b}"
