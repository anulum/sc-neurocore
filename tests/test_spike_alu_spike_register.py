# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeRegister from former test_spike_alu.py

"""Focused suite: TestSpikeRegister from former test_spike_alu.py."""

from __future__ import annotations

from tests.spike_alu_support import *  # noqa: F403

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
