# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for spike-based ALU (Turing-complete computation)

"""Tests for SpikeGate, SpikeRegister, SpikeALU, spike_sort."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.symbolic.spike_logic import (
    SpikeGate,
    SpikeRegister,
    SpikeALU,
    spike_sort,
)


class TestSpikeGate:
    @pytest.mark.parametrize("a,b,expected", [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)])
    def test_and_truth_table(self, a, b, expected):
        assert SpikeGate("AND")(a, b) == expected

    @pytest.mark.parametrize("a,b,expected", [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
    def test_or_truth_table(self, a, b, expected):
        assert SpikeGate("OR")(a, b) == expected

    @pytest.mark.parametrize("a,expected", [(0, 1), (1, 0)])
    def test_not_truth_table(self, a, expected):
        assert SpikeGate("NOT")(a) == expected

    @pytest.mark.parametrize("a,b,expected", [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)])
    def test_nand_truth_table(self, a, b, expected):
        assert SpikeGate("NAND")(a, b) == expected

    @pytest.mark.parametrize("a,b,expected", [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)])
    def test_xor_truth_table(self, a, b, expected):
        assert SpikeGate("XOR")(a, b) == expected

    def test_lif_config_exists(self):
        for gate_type in ["AND", "OR", "NOT", "NAND", "XOR"]:
            gate = SpikeGate(gate_type)
            config = gate.lif_config
            assert isinstance(config, dict)

    def test_de_morgan_and(self):
        """NOT(A AND B) = NOT(A) OR NOT(B)."""
        nand = SpikeGate("NAND")
        not_g = SpikeGate("NOT")
        or_g = SpikeGate("OR")
        for a in [0, 1]:
            for b in [0, 1]:
                assert nand(a, b) == or_g(not_g(a), not_g(b))


class TestSpikeRegister:
    def test_write_read_roundtrip(self):
        reg = SpikeRegister(n_bits=8)
        for val in [0, 42, 127, 255]:
            reg.write(val)
            assert reg.read() == val

    def test_clear(self):
        reg = SpikeRegister(n_bits=8)
        reg.write(255)
        reg.clear()
        assert reg.read() == 0

    def test_bit_level_access(self):
        reg = SpikeRegister(n_bits=8)
        reg.write(0b10110011)
        bits = reg.read_bits()
        assert len(bits) == 8
        # LSB first
        expected = np.array([1, 1, 0, 0, 1, 1, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(bits, expected)

    def test_write_bits(self):
        reg = SpikeRegister(n_bits=4)
        reg.write_bits(np.array([1, 0, 1, 0], dtype=np.uint8))
        assert reg.read() == 0b0101  # LSB first

    def test_16_bit(self):
        reg = SpikeRegister(n_bits=16)
        reg.write(65535)
        assert reg.read() == 65535
        reg.write(0)
        assert reg.read() == 0


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


class TestSpikeSort:
    def test_empty(self):
        assert spike_sort([]) == []

    def test_single(self):
        assert spike_sort([42]) == [42]

    def test_sorted_input(self):
        assert spike_sort([1, 2, 3, 4]) == [1, 2, 3, 4]

    def test_reversed_input(self):
        assert spike_sort([9, 7, 5, 3, 1]) == [1, 3, 5, 7, 9]

    def test_duplicates(self):
        assert spike_sort([5, 5, 5]) == [5, 5, 5]

    def test_large_values(self):
        arr = [255, 0, 128, 64, 192]
        assert spike_sort(arr) == sorted(arr)

    def test_random_array(self):
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, size=15).tolist()
        assert spike_sort(arr) == sorted(arr)
