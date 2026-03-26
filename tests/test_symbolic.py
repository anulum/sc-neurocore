# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for symbolic spike logic module

import numpy as np
import pytest

from sc_neurocore.symbolic import SpikeGate, SpikeRegister, SpikeALU, spike_sort


class TestSpikeGate:
    @pytest.mark.parametrize(
        "gate,inputs,expected",
        [
            ("AND", (1, 1), 1),
            ("AND", (1, 0), 0),
            ("AND", (0, 1), 0),
            ("AND", (0, 0), 0),
            ("OR", (1, 1), 1),
            ("OR", (1, 0), 1),
            ("OR", (0, 1), 1),
            ("OR", (0, 0), 0),
            ("NOT", (1,), 0),
            ("NOT", (0,), 1),
            ("NAND", (1, 1), 0),
            ("NAND", (1, 0), 1),
            ("NAND", (0, 0), 1),
            ("XOR", (1, 1), 0),
            ("XOR", (1, 0), 1),
            ("XOR", (0, 1), 1),
            ("XOR", (0, 0), 0),
        ],
    )
    def test_truth_tables(self, gate, inputs, expected):
        g = SpikeGate(gate)
        assert g(*inputs) == expected

    def test_xor_three_inputs(self):
        g = SpikeGate("XOR")
        assert g(1, 1, 1) == 1
        assert g(1, 1, 0) == 0
        assert g(1, 0, 0) == 1

    def test_lif_config_keys(self):
        for gate_type in ("AND", "OR", "NOT", "NAND", "XOR"):
            config = SpikeGate(gate_type).lif_config
            assert isinstance(config, dict)
            assert len(config) > 0

    def test_and_gate_lif_threshold(self):
        assert SpikeGate("AND").lif_config["threshold"] == 2

    def test_or_gate_lif_threshold(self):
        assert SpikeGate("OR").lif_config["threshold"] == 1


class TestSpikeRegister:
    def test_write_read_roundtrip(self):
        reg = SpikeRegister(8)
        for val in [0, 1, 127, 255]:
            reg.write(val)
            assert reg.read() == val

    def test_write_bits(self):
        reg = SpikeRegister(4)
        reg.write_bits(np.array([1, 0, 1, 1], dtype=np.int8))
        # bit 0=1, bit 1=0, bit 2=1, bit 3=1 → 0b1101 = 13
        assert reg.read() == 0b1101

    def test_clear(self):
        reg = SpikeRegister(8)
        reg.write(255)
        reg.clear()
        assert reg.read() == 0

    def test_read_bits(self):
        reg = SpikeRegister(4)
        reg.write(0b1010)
        bits = reg.read_bits()
        np.testing.assert_array_equal(bits, [0, 1, 0, 1])

    def test_16bit_register(self):
        reg = SpikeRegister(16)
        reg.write(0xABCD)
        assert reg.read() == 0xABCD


class TestSpikeALU:
    def test_add_basic(self):
        alu = SpikeALU(8)
        result, carry = alu.add(100, 50)
        assert result == 150
        assert carry is False

    def test_add_overflow(self):
        alu = SpikeALU(8)
        result, carry = alu.add(200, 100)
        assert result == (300 & 0xFF)
        assert carry is True

    def test_add_zero(self):
        alu = SpikeALU(8)
        result, carry = alu.add(0, 0)
        assert result == 0
        assert carry is False

    @pytest.mark.parametrize("a,b", [(50, 30), (255, 1), (100, 100), (1, 0)])
    def test_add_matches_python(self, a, b):
        alu = SpikeALU(8)
        result, _ = alu.add(a, b)
        assert result == (a + b) & 0xFF

    def test_sub_basic(self):
        alu = SpikeALU(8)
        result, borrow = alu.sub(100, 30)
        assert result == 70
        assert borrow is False

    def test_sub_underflow(self):
        alu = SpikeALU(8)
        result, borrow = alu.sub(10, 20)
        assert result == (10 - 20) & 0xFF
        assert borrow is True

    def test_bitwise_and(self):
        alu = SpikeALU(8)
        assert alu.bitwise_and(0b11001100, 0b10101010) == 0b10001000

    def test_bitwise_or(self):
        alu = SpikeALU(8)
        assert alu.bitwise_or(0b11001100, 0b10101010) == 0b11101110

    def test_bitwise_xor(self):
        alu = SpikeALU(8)
        assert alu.bitwise_xor(0b11001100, 0b10101010) == 0b01100110

    def test_compare(self):
        alu = SpikeALU(8)
        assert alu.compare(10, 20) == -1
        assert alu.compare(20, 10) == 1
        assert alu.compare(10, 10) == 0

    def test_shift_left(self):
        alu = SpikeALU(8)
        assert alu.shift_left(0b00000001, 3) == 0b00001000
        assert alu.shift_left(0b10000000, 1) == 0

    def test_shift_right(self):
        alu = SpikeALU(8)
        assert alu.shift_right(0b10000000, 3) == 0b00010000
        assert alu.shift_right(1, 1) == 0

    def test_16bit_alu(self):
        alu = SpikeALU(16)
        result, carry = alu.add(30000, 30000)
        assert result == 60000
        assert carry is False


class TestSpikeSorter:
    def test_sort_basic(self):
        assert spike_sort([3, 1, 4, 1, 5, 9, 2, 6]) == sorted([3, 1, 4, 1, 5, 9, 2, 6])

    def test_sort_already_sorted(self):
        assert spike_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]

    def test_sort_reversed(self):
        assert spike_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    def test_sort_single(self):
        assert spike_sort([42]) == [42]

    def test_sort_empty(self):
        assert spike_sort([]) == []

    def test_sort_duplicates(self):
        assert spike_sort([7, 7, 7, 7]) == [7, 7, 7, 7]

    def test_sort_large_values(self):
        vals = [200, 50, 128, 0, 255]
        assert spike_sort(vals) == sorted(vals)
