# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeRegister from former test_symbolic.py

"""Focused suite: TestSpikeRegister from former test_symbolic.py."""

from __future__ import annotations

from tests.symbolic_support import *  # noqa: F403

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
