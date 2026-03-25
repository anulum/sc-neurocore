# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Tests for sc_neurocore.symbolic

from __future__ import annotations
import numpy as np
from sc_neurocore.symbolic import SpikeGate, SpikeRegister, SpikeALU, spike_sort


class TestSpikeGate:
    def test_and(self):
        g = SpikeGate("AND")
        assert g(1, 1) == 1
        assert g(1, 0) == 0
        assert g(0, 0) == 0

    def test_or(self):
        g = SpikeGate("OR")
        assert g(0, 0) == 0
        assert g(1, 0) == 1
        assert g(1, 1) == 1

    def test_not(self):
        g = SpikeGate("NOT")
        assert g(0) == 1
        assert g(1) == 0

    def test_nand(self):
        g = SpikeGate("NAND")
        assert g(1, 1) == 0
        assert g(1, 0) == 1

    def test_xor(self):
        g = SpikeGate("XOR")
        assert g(0, 0) == 0
        assert g(1, 0) == 1
        assert g(0, 1) == 1
        assert g(1, 1) == 0

    def test_lif_config(self):
        g = SpikeGate("AND")
        cfg = g.lif_config
        assert cfg["threshold"] == 2


class TestSpikeRegister:
    def test_write_read(self):
        r = SpikeRegister(8)
        r.write(42)
        assert r.read() == 42

    def test_clear(self):
        r = SpikeRegister(8)
        r.write(255)
        r.clear()
        assert r.read() == 0

    def test_bits(self):
        r = SpikeRegister(4)
        r.write_bits(np.array([1, 0, 1, 0], dtype=np.int8))
        assert r.read() == 5  # binary 0101 = 5


class TestSpikeALU:
    def test_add(self):
        alu = SpikeALU(8)
        result, carry = alu.add(10, 20)
        assert result == 30
        assert not carry

    def test_add_overflow(self):
        alu = SpikeALU(8)
        result, carry = alu.add(200, 100)
        assert result == (300 & 0xFF)  # 44
        assert carry

    def test_sub(self):
        alu = SpikeALU(8)
        result, borrow = alu.sub(30, 10)
        assert result == 20
        assert not borrow

    def test_sub_underflow(self):
        alu = SpikeALU(8)
        _, borrow = alu.sub(10, 30)
        assert borrow

    def test_bitwise_and(self):
        alu = SpikeALU(8)
        assert alu.bitwise_and(0b1100, 0b1010) == 0b1000

    def test_bitwise_or(self):
        alu = SpikeALU(8)
        assert alu.bitwise_or(0b1100, 0b1010) == 0b1110

    def test_bitwise_xor(self):
        alu = SpikeALU(8)
        assert alu.bitwise_xor(0b1100, 0b1010) == 0b0110

    def test_compare(self):
        alu = SpikeALU(8)
        assert alu.compare(10, 20) == -1
        assert alu.compare(20, 10) == 1
        assert alu.compare(10, 10) == 0

    def test_shift(self):
        alu = SpikeALU(8)
        assert alu.shift_left(1, 3) == 8
        assert alu.shift_right(8, 2) == 2


class TestSpikeSort:
    def test_basic(self):
        assert spike_sort([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]

    def test_already_sorted(self):
        assert spike_sort([1, 2, 3]) == [1, 2, 3]

    def test_reverse(self):
        assert spike_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    def test_single(self):
        assert spike_sort([42]) == [42]

    def test_empty(self):
        assert spike_sort([]) == []
