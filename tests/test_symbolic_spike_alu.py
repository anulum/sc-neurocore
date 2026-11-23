# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeALU from former test_symbolic.py

"""Focused suite: TestSpikeALU from former test_symbolic.py."""

from __future__ import annotations

import pytest
from sc_neurocore.symbolic import (
    SpikeALU,
)


class TestSpikeALU:
    """The spiking arithmetic logic unit."""

    def test_add_basic(self) -> None:
        """100 + 50 gives 150 without carry."""
        alu = SpikeALU(8)
        result, carry = alu.add(100, 50)
        assert result == 150
        assert carry is False

    def test_add_overflow(self) -> None:
        """An 8-bit overflow wraps and sets the carry."""
        alu = SpikeALU(8)
        result, carry = alu.add(200, 100)
        assert result == (300 & 0xFF)
        assert carry is True

    def test_add_zero(self) -> None:
        """0 + 0 gives 0 without carry."""
        alu = SpikeALU(8)
        result, carry = alu.add(0, 0)
        assert result == 0
        assert carry is False

    @pytest.mark.parametrize("a,b", [(50, 30), (255, 1), (100, 100), (1, 0)])
    def test_add_matches_python(self, a: int, b: int) -> None:
        """Addition matches Python modulo 256."""
        alu = SpikeALU(8)
        result, _ = alu.add(a, b)
        assert result == (a + b) & 0xFF

    def test_sub_basic(self) -> None:
        """100 - 30 gives 70 without borrow."""
        alu = SpikeALU(8)
        result, borrow = alu.sub(100, 30)
        assert result == 70
        assert borrow is False

    def test_sub_underflow(self) -> None:
        """An underflow wraps and sets the borrow."""
        alu = SpikeALU(8)
        result, borrow = alu.sub(10, 20)
        assert result == (10 - 20) & 0xFF
        assert borrow is True

    def test_bitwise_and(self) -> None:
        """Bitwise AND matches the truth table."""
        alu = SpikeALU(8)
        assert alu.bitwise_and(0b11001100, 0b10101010) == 0b10001000

    def test_bitwise_or(self) -> None:
        """Bitwise OR matches the truth table."""
        alu = SpikeALU(8)
        assert alu.bitwise_or(0b11001100, 0b10101010) == 0b11101110

    def test_bitwise_xor(self) -> None:
        """Bitwise XOR matches the truth table."""
        alu = SpikeALU(8)
        assert alu.bitwise_xor(0b11001100, 0b10101010) == 0b01100110

    def test_compare(self) -> None:
        """Comparison gives -1, 1 or 0."""
        alu = SpikeALU(8)
        assert alu.compare(10, 20) == -1
        assert alu.compare(20, 10) == 1
        assert alu.compare(10, 10) == 0

    def test_shift_left(self) -> None:
        """A left shift drops the high bits."""
        alu = SpikeALU(8)
        assert alu.shift_left(0b00000001, 3) == 0b00001000
        assert alu.shift_left(0b10000000, 1) == 0

    def test_shift_right(self) -> None:
        """A right shift drops the low bits."""
        alu = SpikeALU(8)
        assert alu.shift_right(0b10000000, 3) == 0b00010000
        assert alu.shift_right(1, 1) == 0

    def test_16bit_alu(self) -> None:
        """A 16-bit ALU adds 30000 + 30000 without carry."""
        alu = SpikeALU(16)
        result, carry = alu.add(30000, 30000)
        assert result == 60000
        assert carry is False
